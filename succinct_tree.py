import math
import sdsl4py
import numpy as np

from rmm_tree import BucketRMMinMax

#------------------------------------------------------------------------------
# SuccinctTree
#------------------------------------------------------------------------------

class SuccinctTree:
    """
    Succinct tree over a BP string, supporting all navigation operations
    in O(log log n) time and 2n+O(n/log n) bits of space.
    """
    def __init__(self, bp_string: str, beta: int = 1 << 15, block_size: int = 1024):
        # Need to check that input is actually a single tree and not a forest?

        # ------------------------------------------------------------
        # 0. store raw bit-vector ( ( =1, ) =0 )
        # ------------------------------------------------------------
        self.n2 = len(bp_string)
        self.beta = beta
        self.b = block_size

        # # Adaptive bucket and block sizes
        # logn    = max(1.0, math.log2(self.n2))
        # loglogn = max(1.0, math.log2(logn))
        # self.beta       = max(32, int(math.ceil(logn**3)))          # Θ((log n)³)
        # self.b = max(16, int(math.ceil(logn * loglogn)))            # Θ(log n·log log n)
        # B = (self.n2 + self.beta - 1)//self.beta                    # O(n/(log n)³) buckets

        # Empty tree guard
        if self.n2 == 0:
            self.bv    = []
            return

        self.bv = sdsl4py.bit_vector(size=self.n2, default_value=0)
        for i, ch in enumerate(bp_string):
            self.bv[i] = (ch == '(')

        # For testing
        # B aliases bv, so we can remove B
        self.B = np.array(self.bv, dtype=np.uint8)
        self._names   = [None] * self.n2
        self._lengths = [0.0]  * self.n2

        # ------------------------------------------------------------
        # 1. build buckets
        # ------------------------------------------------------------
        self.num_buckets = math.ceil(self.n2 / self.beta)
        self.buckets = []
        for bi in range(self.num_buckets):
            lo = bi * self.beta
            hi = min((bi + 1) * self.beta, self.n2)
            try:
                bits = self.bv[lo:hi]           # slice if the wrapper supports it
            except TypeError:
                bits = [bool(self.bv[i]) for i in range(lo, hi)]
            self.buckets.append(BucketRMMinMax(bits, self.b))

        # ------------------------------------------------------------
        # 2. per-bucket absolute excess @ END of bucket
        # ------------------------------------------------------------
        self.bucket_excess = []
        acc = 0
        for bk in self.buckets:
            acc += bk._excess_local(bk.n - 1) if bk.n else 0
            self.bucket_excess.append(acc)

        # ------------------------------------------------------------
        # 3. absolute minima / maxima of each bucket
        #    bucket_m[k]  = global excess minimum inside bucket k
        # ------------------------------------------------------------
        self.bucket_m, self.bucket_M = [], []
        for k, bk in enumerate(self.buckets):
            _, m_rel, M_rel, _ = bk._combine_range(0, len(bk.leaf_summaries) - 1)
            base = self.bucket_excess[k - 1] if k > 0 else 0
            self.bucket_m.append(base + m_rel)
            self.bucket_M.append(base + M_rel)

        # ------------------------------------------------------------
        # 4. prev-smaller forest  +  ladder decomposition
        # ------------------------------------------------------------
        B = self.num_buckets
        
        self.prev_smaller = ps = [-1] * B
        st = []
        for k, val in enumerate(self.bucket_m):
            while st and self.bucket_m[st[-1]] > val:
                st.pop()
            ps[k] = st[-1] if st else -1
            st.append(k)

        # heavy-path heads and depths
        self.head  = head  = [None] * B
        self.depth_arr = depth_arr = [0] * B
        for k in range(B):
            if head[k] is not None:
                continue
            path = []
            x = k
            while x != -1 and head[x] is None:
                path.append(x)
                x = ps[x]
            base_depth = depth_arr[x] + 1 if x != -1 else 0
            # topmost node of this heavy path
            path_head = path[-1]
            for idx, y in enumerate(reversed(path)):
                head[y]  = path_head   # every node points to the *same* head
                depth_arr[y] = base_depth + idx

        # binary-lifting table
        max_ladders = max(1, math.ceil(math.log2(B)))     # ≤ ⌈log₂B⌉ = O(log n)

        # only need ⌈log₂(max_ladders)⌉ levels
        L = max(1, math.ceil(math.log2(max_ladders)))  

        # allocate exactly L rows
        self.up = up = [[-1] * B for _ in range(L)]

        # initialize 0th row to your prev-smaller parents
        up[0] = ps

        # fill the rest as before
        for j in range(1, L):
            prev = up[j - 1]
            row  = up[j]
            for k in range(B):
                p = prev[k]
                row[k] = prev[p] if p != -1 else -1

        # ------------------------------------------------------------
        # 5. perfect binary tree on buckets: L/R prefix minima & maxima
        # ------------------------------------------------------------
        self.bucket_summaries = [
            bk._combine_range(0, len(bk.leaf_summaries) - 1)
            for bk in self.buckets
        ]

        P = 1 << math.ceil(math.log2(B or 1))
        size = 2 * P
        self.P = P
        self.Lmin = Lmin = [1 << 30] * size
        self.Rmin = Rmin = [1 << 30] * size
        self.Lmax = Lmax = [-(1 << 30)] * size
        self.Rmax = Rmax = [-(1 << 30)] * size
        self.CntL = CntL = [0] * size   # minima counts on left prefixes
        self.CntR = CntR = [0] * size   # minima counts on right suffixes

        # leaves
        for k, (mn, mx) in enumerate(zip(self.bucket_m, self.bucket_M)):
            idx = P + k
            Lmin[idx] = Rmin[idx] = mn
            Lmax[idx] = Rmax[idx] = mx
            nk = self.bucket_summaries[k][3]            # n_k of the whole bucket
            CntL[idx] = CntR[idx] = nk

        # bottom-up combine
        for v in range(P - 1, 0, -1):
            l, r = 2 * v, 2 * v + 1
            # minima
            Lmin[v] = min(Lmin[l], Lmin[r])
            Rmin[v] = min(Rmin[r], Rmin[l])
            CntL[v] = (CntL[l] if Lmin[l] == Lmin[v] else 0) + \
                      (CntL[r] if Lmin[r] == Lmin[v] else 0)
            CntR[v] = (CntR[r] if Rmin[r] == Rmin[v] else 0) + \
                      (CntR[l] if Rmin[l] == Rmin[v] else 0)
            # maxima
            Lmax[v] = max(Lmax[l], Lmax[r])
            Rmax[v] = max(Rmax[r], Rmax[l])

        # ------------------------------------------------------------
        # 8. rank / select (sdsl4py O(1))
        # ------------------------------------------------------------
        self.rs1 = sdsl4py.RankSupportV1(self.bv)
        self.ss1 = sdsl4py.SelectSupportMCL1(self.bv)
        self.rs0 = sdsl4py.RankSupportV0(self.bv)
        self.ss0 = sdsl4py.SelectSupportMCL0(self.bv)

    # For testing
    def name(self, i: int):
        """Return the stored name at node i, or None if unset/out-of-bounds."""
        if i < 0 or i >= self.n2:
            return None
        return self._names[i]

    def length(self, i: int) -> float:
        """Return the stored length at node i, or 0.0 if unset/out-of-bounds."""
        if i < 0 or i >= self.n2:
            return 0.0
        return self._lengths[i]

    # ------------------------------------------------------------
    # helper 1 – first bucket ≥ k whose min ≤ t   (O(1) with ladder)
    # ------------------------------------------------------------
    def _first_bucket_leq(self, k: int, t: int):
        B = self.num_buckets
        if k >= B:
            return None
        if self.bucket_m[k] <= t:
            return k
        # jump to head of heavy path
        k = self.head[k]
        # binary-lift while min still > t
        for j in reversed(range(len(self.up))):
            nxt = self.up[j][k]
            if nxt != -1 and self.bucket_m[nxt] > t:
                k = nxt
        k = self.up[0][k]          # final ≤ 1 step
        return k if k != -1 and self.bucket_m[k] <= t else None
    
    def _last_bucket_leq(self, k: int, tgt: int):
        """
        Return the right-most bucket ≤ k whose M (max excess)  ≥ tgt.
        Uses the prev-smaller ladder + binary-lifting.
        """
        while k is not None and k >= 0 and self.bucket_M[k] < tgt:
            k = self.head[k]                # jump to path head
            # binary-lift leftwards along ladder
            for j in reversed(range(len(self.up))):
                nxt = self.up[j][k]
                if nxt != -1 and self.bucket_M[nxt] < tgt:
                    k = nxt
            k = self.up[0][k]               # final ≤1 step
        return k


    # ------------------------------------------------------------
    # helper 2 – first bucket with global min in (i..j)
    #            uses L/R prefix arrays  (O(1))
    # ------------------------------------------------------------
    def _rmq_bucket_range(self, i: int, j: int):
        if i > j:
            return None
        
        P   = self.P
        v_l = i + P
        v_r = j + P
        min_left  =  1 << 30; bucket_left  = None
        min_right =  1 << 30; bucket_right = None
        while v_l <= v_r:
            if v_l & 1:
                if self.Lmin[v_l] < min_left:
                    min_left, bucket_left = self.Lmin[v_l], v_l
                v_l += 1
            if not (v_r & 1):
                if self.Rmin[v_r] < min_right:
                    min_right, bucket_right = self.Rmin[v_r], v_r
                v_r -= 1
            v_l >>= 1; v_r >>= 1
        # decide earlier bucket
        if min_left < min_right:    
            v = bucket_left
            while v < P:
                v <<= 1
                if self.Lmin[v] != min_left:
                    v += 1
            return v - P
        else:
            v = bucket_right
            while v < P:
                v = v * 2 + 1
                if self.Rmin[v] != min_right:
                    v -= 1
            return v - P
        
    def _rMq_bucket_range(self, i: int, j: int):
        """Return first bucket index in (i..j) with the global MAX excess."""
        if i > j:
            return None
        
        P = self.P
        v_l, v_r = i + P, j + P
        max_left, max_right = -(1 << 30), -(1 << 30)
        bucket_left = bucket_right = None
        while v_l <= v_r:
            if v_l & 1:
                if self.Lmax[v_l] > max_left:
                    max_left, bucket_left = self.Lmax[v_l], v_l
                v_l += 1
            if not (v_r & 1):
                if self.Rmax[v_r] > max_right:
                    max_right, bucket_right = self.Rmax[v_r], v_r
                v_r -= 1
            v_l >>= 1; v_r >>= 1
        if max_left > max_right:     
            v = bucket_left
            while v < P:
                v <<= 1
                if self.Lmax[v] != max_left:
                    v += 1
            return v - P
        else:
            v = bucket_right
            while v < P:
                v = v * 2 + 1
                if self.Rmax[v] != max_right:
                    v -= 1
            return v - P
    
    # ------------------------------------------------------------
    # bucket-range helpers for mincount / minselect
    # ------------------------------------------------------------
    def _mincount_bucket_range(self, i: int, j: int, target: int) -> int:
        """Count buckets whose minimum == target in [i..j] in O(log log n)."""
        if i > j:
            return 0
        P, cnt = self.P, 0
        v_l, v_r = i + P, j + P
        while v_l <= v_r:
            if v_l & 1:                          # v_l is a right child
                if self.Lmin[v_l] == target:
                    cnt += self.CntL[v_l]
                v_l += 1
            if not (v_r & 1):                    # v_r is a left child
                if self.Rmin[v_r] == target:
                    cnt += self.CntR[v_r]
                v_r -= 1
            v_l >>= 1
            v_r >>= 1
        return cnt

    def _minselect_bucket_range(self, i: int, j: int,
                                target: int, q: int) -> int:
        """
        Return the bucket index of the q-th occurrence of `target`
        in [i..j] (1-based q).  Assumes q ≤ total count.
        """
        P = self.P
        v_l, v_r = i + P, j + P
        # first, collect the right traversals to replay later
        rights = []
        while v_l <= v_r:
            if v_l & 1:
                rights.append(v_l)
                v_l += 1
            if not (v_r & 1):
                v_r -= 1
            v_l >>= 1
            v_r >>= 1
        # now replay the saved right children in order
        for v in rights:
            if self.Lmin[v] == target:
                if q <= self.CntL[v]:
                    # descend to the exact bucket
                    while v < P:
                        v <<= 1
                        if self.Lmin[v] != target:
                            v += 1
                    return v - P
                q -= self.CntL[v]
        # we should never reach here if inputs are consistent
        return None


    #-------------------------------------------------------------------------
    # Basic bitvector support
    #-------------------------------------------------------------------------
    def rank1(self, pos: int) -> int:
        """
        #1s in [0..pos), i.e. up to but excluding pos.
        """
        pos = min(max(pos, 0), self.n2)
        # SDSL rank(x) returns #1s in [0..x)
        return self.rs1.rank(pos)

    def select1(self, k: int):
        """
        0-based: index of the k-th '(', or None if out of range.
        """
        total_ones = self.rs1.rank(self.n2)   # total 1-bits
        if k < 0 or k >= total_ones:
            return None
        # SDSL select is 1-based: select(1)→first '(', so pass k+1
        return self.ss1.select(k+1)

    def rank0(self, pos: int) -> int:
        """
        #0s in [0..pos), i.e. up to but excluding pos.
        """
        pos = min(max(pos, 0), self.n2)
        return self.rs0.rank(pos)

    def select0(self, k: int):
        """
        0-based: index of the k-th ')', or None if out of range.
        """
        total_zeros = self.rs0.rank(self.n2)  # total 0-bits
        if k < 0 or k >= total_zeros:
            return None
        return self.ss0.select(k+1)

    #-------------------------------------------------------------------------
    # Core primitives
    #-------------------------------------------------------------------------
    def excess(self, i: int):
        if i < 0:
            return 0
        if i >= self.n2:
            i = self.n2 - 1

        bi, off = divmod(i, self.beta)
        before  = self.bucket_excess[bi - 1] if bi else 0
        intra   = self.buckets[bi]._excess_local(off)   
        return before + intra
    

    def rmq(self, i, j):
        if i > j:
            i, j = j, i

        bi, bj = i // self.beta, j // self.beta
        offi, offj = i % self.beta, j % self.beta

        # ---- whole range inside one bucket ----------------------------
        if bi == bj:
            p, _ = self.buckets[bi].rmq_in_bucket(offi, offj)
            return bi * self.beta + p

        # ---- general (≥ 2 buckets) ------------------------------------
        best_val = float("inf")

        # first bucket
        p, v = self.buckets[bi].rmq_in_bucket(offi, self.buckets[bi].n - 1)
        best_val, best_pos = v, bi * self.beta + p

        # last bucket
        p2, v2 = self.buckets[bj].rmq_in_bucket(0, offj)
        if v2 < best_val:
            best_val, best_pos = v2, bj * self.beta + p2

        # middle buckets
        if bj > bi + 1:
            k = self._rmq_bucket_range(bi + 1, bj - 1)
            if k is not None and self.bucket_m[k] <= best_val:
                rel = self.bucket_m[k] - (self.bucket_excess[k - 1] if k else 0)
                p3 = self.buckets[k].find_first_excess(rel)
                if p3 is not None and (
                    self.bucket_m[k] < best_val
                    or (self.bucket_m[k] == best_val and k * self.beta + p3 < best_pos)
                ):
                    best_pos = k * self.beta + p3

        return best_pos


    def rMq(self, i, j):
        if i > j:
            i, j = j, i

        bi, bj   = i // self.beta, j // self.beta
        offi, offj = i % self.beta, j % self.beta

        # ---- whole query inside ONE bucket ----------------------------
        if bi == bj:
            p, _ = self.buckets[bi].rMq_in_bucket(offi, offj)
            return bi * self.beta + p

    # ---- general case ( ≥ 2 buckets ) -----------------------------

        # ---------- first bucket ----------------------------------------
        p, v = self.buckets[bi].rMq_in_bucket(offi, self.buckets[bi].n - 1)
        best_val = v
        best_pos = bi * self.beta + p

        # ---------- last bucket -----------------------------------------
        if bj != bi:
            p2, v2 = self.buckets[bj].rMq_in_bucket(0, offj)
            if v2 > best_val:
                best_val = v2           # <- no trailing comma!
                best_pos = bj * self.beta + p2

        # ---------- middle buckets --------------------------------------
        if bj > bi + 1:
            k = self._rMq_bucket_range(bi + 1, bj - 1)
            if k is not None and self.bucket_M[k] > best_val:
                relM = self.bucket_M[k] - (self.bucket_excess[k - 1] if k else 0)
                p3 = self.buckets[k].find_last_excess(relM)
                if p3 is not None:
                    best_pos = k * self.beta + p3

        return best_pos

    # ------------------------------------------------------------
    # forward search : first j > i with excess(j) == excess(i)+d
    # ------------------------------------------------------------
    def fwdsearch(self, i: int, d: int):
        if i < 0 or i >= self.n2:
            return None

        tgt = self.excess(i) + d
        bi, off = divmod(i, self.beta)

        # 1) finish current bucket
        p = self.buckets[bi].fwdsearch_in_bucket(off, tgt)
        if p is not None:
            return bi * self.beta + p

        # 2) cross buckets via ladder
        k = self._first_bucket_leq(bi + 1, tgt)
        while k is not None and self.bucket_M[k] < tgt:      # min ok, max too small
            k = self._first_bucket_leq(k + 1, tgt)

        if k is not None:
            p2 = self.buckets[k].fwdsearch_in_bucket(0, tgt)
            if p2 is not None:
                return k * self.beta + p2
        return None

    # ------------------------------------------------------------
    # backward search : last j < i with excess(j) == excess(i)+d
    # ------------------------------------------------------------
    def bwdsearch(self, i: int, d: int):
        """
        Largest j < i with  excess(j) == excess(i) + d.
        Runs in  O(log log n)  worst-case:
        • ≤ b bits in current bucket
        • one heavy-path jump  +  ≤ log β binary-lift steps.
        """
        # clamp i into [0..n2-1]
        if i < 0:
            return None
        if i >= self.n2:
            i = self.n2 - 1   

        tgt = self.excess(i) + d
        bi, off = divmod(i, self.beta)

        # If they're asking for target=0, the 'sentinel' at -1 matches:
        if tgt == 0:
            return -1

        # -------------------------------------------------------------
        # 1) scan inside the current bucket
        # -------------------------------------------------------------
        p = self.buckets[bi].bwdsearch_in_bucket(off, tgt)
        if p is not None:
            return bi * self.beta + p

        # -------------------------------------------------------------
        # 2) skip buckets whose *minimum* is still  > tgt
        #    (prev_smaller hops along the ladder, O(log log n))
        # -------------------------------------------------------------
        k = bi - 1
        while k != -1 and self.bucket_m[k] > tgt:
            k = self.prev_smaller[k]

        if k == -1:
            return None           # no bucket can contain the target

        # -------------------------------------------------------------
        # 3) hop to the right-most bucket ≤ k whose  max ≥ tgt
        #    using ladder head + binary-lifting  (O(log log n))
        # -------------------------------------------------------------
        k = self._last_bucket_leq(k, tgt)     # helper defined earlier
        if k is None or k < 0:
            return None

        # -------------------------------------------------------------
        # 4) final in-bucket backward search (≤ b bits)
        # -------------------------------------------------------------
        p2 = self.buckets[k].bwdsearch_in_bucket(self.buckets[k].n - 1, tgt)
        return k * self.beta + p2 if p2 is not None else None

    def _mincount_naive(self, i, j, target):
        """Scan excess from i→j in O(j-i) time."""
        cnt = 0
        cur = self.excess(i)
        # we assume j≥i
        for pos in range(i+1, j+1):
            cur += 1 if self.bv[pos] else -1
            if cur == target:
                cnt += 1
        return cnt

    def _minselect_naive(self, i, j, target, q):
        """Find q-th position of 'target' in excess[i→j] in O(j-i) time."""
        # start the running excess *just before* i
        cur = self.excess(i-1) if i > 0 else 0

        # scan every position from i up through j, inclusive
        for pos in range(i, j+1):
            cur += 1 if self.bv[pos] else -1
            if cur == target:
                q -= 1
                if q == 0:
                    return pos
        return None
    
    def mincount(self, i, j):
        """Number of times the minimum excess appears in [i..j]."""
        if i > j:
            i, j = j, i

        tgt_pos = self.rmq(i, j)
        target  = self.excess(tgt_pos)

        # ---------- single bucket --------------------------------------
        bi, bj   = i // self.beta, j // self.beta
        offi, offj = i % self.beta, j % self.beta
        if bi == bj:
            return self.buckets[bi].mincount_in_bucket(offi, offj, target)

        # ---------- small range entirely within β bits -----------------
        if j - i <= self.beta:
            return self._mincount_naive(i, j, target)

        # ---------- general case (≥ 2 buckets) -------------------------
        cnt  = self.buckets[bi].mincount_in_bucket(offi,
                                                self.buckets[bi].n - 1,
                                                target)
        cnt += self.buckets[bj].mincount_in_bucket(0, offj, target)

        if bj > bi + 1:
            cnt += self._mincount_bucket_range(bi + 1, bj - 1, target)
        return cnt

    def minselect(self, i, j, q):
        """Position of the q-th minimum in excess[i..j]."""
        if i > j:
            i, j = j, i

        tgt_pos = self.rmq(i, j)
        target  = self.excess(tgt_pos)

        # ---------- single bucket --------------------------------------
        bi, bj   = i // self.beta, j // self.beta
        offi, offj = i % self.beta, j % self.beta
        
        if bi == bj:
            local = self.buckets[bi].minselect_in_bucket(offi, offj, target, q)
            return None if local is None else bi * self.beta + local

        # ---------- small range ----------------------------------------
        if j - i <= self.beta:
            return self._minselect_naive(i, j, target, q)

        # ---------- general case (≥ 2 buckets) -------------------------
        left_cnt = self.buckets[bi].mincount_in_bucket(offi,
                                                    self.buckets[bi].n - 1,
                                                    target)
        if q <= left_cnt:
            p = self.buckets[bi].minselect_in_bucket(
                offi, self.buckets[bi].n - 1, target, q)
            if p is None:
                return None
            return bi * self.beta + p
        q -= left_cnt

        if bj > bi + 1:
            total = self._mincount_bucket_range(bi + 1, bj - 1, target)
            if q <= total:
                k = self._minselect_bucket_range(bi + 1, bj - 1, target, q)
                if k is None:
                    return None
                prev = self._mincount_bucket_range(bi + 1, k - 1, target)
                q   -= prev
                p = self.buckets[k].minselect_in_bucket(
                    0, self.buckets[k].n - 1, target, q)
                if p is None:
                    return None
                return k * self.beta + p
            q -= total

        right_cnt = self.buckets[bj].mincount_in_bucket(0, offj, target)
        if q <= right_cnt:
            p = self.buckets[bj].minselect_in_bucket(
                0, offj, target, q)
            if p is None:
                return None
            return bj * self.beta + p
        return None            # q out of range

    # ----------------------------------------------------------------------
    #   Fast excess-count helper  (O(log log n))
    # ----------------------------------------------------------------------

    def count_excess(self, i: int, j: int, target: int) -> int:
        """
        Count positions p ∈ [i..j] with excess(p) == *target*.
        O(log log n) worst-case.
        """
        if i > j:
            i, j = j, i
        bi, bj = i // self.beta, j // self.beta
        offi, offj = i % self.beta, j % self.beta

        # (1) left partial bucket
        cnt = self.buckets[bi].mincount_in_bucket(offi,
                                                self.buckets[bi].n - 1,
                                                target)

        # (2) right partial bucket
        if bj != bi:
            cnt += self.buckets[bj].mincount_in_bucket(0, offj, target)

        # (3) full buckets in between
        if bj > bi + 1:
            cnt += self._mincount_bucket_range(bi + 1, bj - 1, target)
        return cnt

    # #-------------------------------------------------------------------------
    # # Matching and core tree operations
    # #-------------------------------------------------------------------------    
    def close(self, i: int):
        """
        Return the index of the ')' matching the '(' at position *i*,
        or None if *i* is out of range or not a '('.
        """
        if i < 0 or i >= self.n2 or not self.bv[i]:
            return None
        return self.fwdsearch(i, -1)                   # excess(i)-1

    def open(self, i: int):
        """
        Return the position of the '(' that matches the ')' at index i.
        If B[i] is not a ')' (or i is out of range) return None.
        """
        if i < 0 or i >= self.n2 or self.bv[i]:      # not a ')' → no match
            return None

        # pos = self.bwdsearch(i, 0)        # last u < i with excess(u) == excess(i)
        pos = self.bwdsearch(i, 0)   # (i, 0) or (i, 1)
        return None if pos is None else pos + 1   # pos + 1 or pos

    def enclose(self, i: int):
        """
        Return the index of the '(' that immediately encloses the '(' at index i,
        or None if i is not an opening parenthesis or is the root.
        """
        # must be a '(' to have an enclosing opener
        if i < 0 or i >= self.n2 or not self.bv[i]:
            return None

        # search for excess == excess(i) - 2
        j = self.bwdsearch(i, -2)
        return None if j is None else j + 1

    def preorder(self, i: int) -> int:
        """
        1-based preorder number of the node whose parenthesis is at index i.
        Accepts either an opener or a closer and returns the same value.
        """
        # 1) bounds check
        if i < 0 or i >= self.n2:
            return None

        # 2) if it’s a closing parenthesis, map it back to its opener
        if not self.bv[i]:
            i = self.open(i)
            if i is None:
                return None

        # 3) rank1(i+1) counts the number of '(' in [0..i], giving 1-based preorder
        return self.rank1(i + 1)

    def postorder(self, i: int) -> int:
        """
        1-based postorder number of the node at opener i (or its matching opener if you pass a closer).
        """
        # 1) if they passed a closer, map to its matching opener
        if i < 0 or i >= self.n2:
            return None
        if not self.bv[i]:
            i = self.open(i)
            if i is None:
                return None

        # 2) find its matching close
        c = self.close(i)
        if c is None:
            return None

        # 3) postorder number = # of zeros in [0..c] inclusive → rank0(c+1)
        return self.rank0(c + 1)

    def preorderselect(self, k: int) -> int:
        """Opener index of the kᵗʰ node in preorder."""
        return self.select1(k)
    
    def postorderselect(self, k: int) -> int:
        """Opener‐index of the kᵗʰ node in postorder (1-based), or None."""
        if k <= 0:
            return None
        # select0 is 0-based (select0(0)→first zero), so use k-1
        pos = self.select0(k - 1)
        return None if pos is None else self.open(pos)
    
    def isleaf(self, i: int) -> bool:
        """True iff the node at i has no children."""
        return i+1 < self.n2 and self.bv[i] and not self.bv[i+1]

    def isancestor(self, i: int, j: int) -> bool:
        """True iff opener i encloses position j."""
        c = self.close(i)
        return (c is not None) and (i < j < c)

    def depth(self, i: int) -> int:
        """
        Tree-depth of the node at position i (root depth = 0).
        Accepts either '(' or ')' indices: maps a closer back to its opener.
        """
        # clamp out of bounds
        if i < 0 or i >= self.n2:
            return 0
        # if it's a closing parenthesis, map to its matching opener
        if not self.bv[i]:
            i = self.open(i)
            if i is None:
                return 0
        # now i is an opener: depth = excess(i) - 1
        return self.excess(i) - 1

    def parent(self, i: int):
        """Opener index of the parent of i, or None."""
        return self.enclose(i)

    def fchild(self, i: int):
        """Opener index of the first child of i, or None."""
        if self.isleaf(i):
            return None
        return i + 1

    def lchild(self, i: int):
        """Opener index of the last child of i, or None."""
        if self.isleaf(i):
            return None
        c = self.close(i)
        pos = self.bwdsearch(c, 0)
        return None if pos is None else (pos if self.bv[pos] else self.open(pos))

    def nsibling(self, i: int):
        """Opener index of the next sibling of i, or None."""
        c = self.close(i)
        pos = self.fwdsearch(c, 0)
        return None if pos is None else (pos if self.bv[pos] else self.open(pos))
    
    def psibling(self, i: int):
        """Opener index of the previous sibling of i, or None."""
        pos = self.bwdsearch(i, 0)
        return None if pos is None else (pos if self.bv[pos] else self.open(pos))

    def subtree(self, i: int) -> int:
        """
        Number of nodes in the subtree rooted at position i.
        Accepts either an opener or a closer index.
        """
        # 1) bounds check
        if i < 0 or i >= self.n2:
            return 0

        # 2) if it's a closing parenthesis, map it back to the opener
        if not self.bv[i]:
            i = self.open(i)
            if i is None:
                return 0

        # 3) find the matching ')'
        c = self.close(i)
        if c is None:
            return 0

        # 4) subtree size = # of '(' between [i..c], inclusive
        #    = rank1(c+1) - rank1(i)
        return self.rank1(c + 1) - self.rank1(i)

    def lca(self, i: int, j: int):
        # ensure i ≤ j
        if i > j:
            i, j = j, i

        # 1) trivial ancestor cases
        if self.isancestor(i, j):
            return i
        if self.isancestor(j, i):
            return j

        # 2) find the position of the minimum excess
        p = self.rmq(i, j)
        if p is None:
            return None

        # 3) the paper says lca = enclose(p + 1)
        return self.enclose(p + 1)

    def root(self):
        """Opener index of the root (first '(')."""
        # return self.select1(1)
        return 0

    def levelancestor(self, i: int, d: int) -> int:
        """
        Return the opener index of the d-th ancestor above i.
        If d=0, returns -1. If you climb above the root, returns the root (0).
        Accepts either '(' or ')' positions.
        """
        # 1) clamp out of bounds
        if i < 0 or i >= self.n2:
            return -1

        # 2) map a closing parenthesis back to its opener
        if not self.bv[i]:
            i = self.open(i)
            if i is None:
                return -1

        # 3) zero‐distance means “no ancestor”
        if d == 0:
            return -1

        # 4) climb up one level at a time
        pos = i
        for _ in range(d):
            pos = self.enclose(pos)
            if pos is None:
                # went above the root → return root (at index 0)
                return 0

        return pos

    def levelnext(self, i: int) -> int:
        """
        Opener index of the next node at the same depth as node i,
        or -1 if none exists.  Runs in O(log log n) worst-case.
        """
        # 1) bounds check
        if i < 0 or i >= self.n2:
            return -1

        # 2) if it's a ')' map it back to its matching '('
        if not self.bv[i]:
            i = self.open(i)
            if i is None:
                return -1

        # 3) get the target excess (depth+1)
        target = self.excess(i)

        # 4) jump to the next position j>i with excess==target
        j = self.fwdsearch(i, 0)
        # 5) skip any closers until we hit an opener (or run out)
        while j is not None and not self.bv[j]:
            j = self.fwdsearch(j, 0)

        return j if j is not None else -1

    def levelprev(self, i: int):
        """Opener index of the previous node on the same depth as i, or None."""
        pos = self.bwdsearch(i, 0)
        if pos is None:
            return None
        return pos if self.bv[pos] else self.open(pos)

    def levelleftmost(self, d: int):
        """Leftmost opener at depth d, or None."""
        if d == 0:
            return self.root()
        # otherwise look for excess == d+1
        pos = self.fwdsearch(self.root(), d+1)
        return None if pos is None else (pos if self.bv[pos] else self.open(pos))

    def levelrightmost(self, d: int):
        """Rightmost opener at depth d, or None."""
        if d == 0:
            return self.root()
        pos = self.bwdsearch(self.n2 - 1, d+1)
        return None if pos is None else (pos if self.bv[pos] else self.open(pos))

    def deepestnode(self, i: int):
        """Opener index of the first deepest node in i’s subtree (accepts openers or closers)."""
        # map a closer back to its opener
        if i < 0 or i >= self.n2:
            return None
        if not self.bv[i]:
            i = self.open(i)
            if i is None:
                return None

        c = self.close(i)
        if c is None:
            return None

        pos = self.rMq(i, c)
        if pos is None:
            return None

        # if pos is a '(', it's already the opener; else map back
        return pos if self.bv[pos] else self.open(pos)

    def height(self, i: int) -> int:
        """Height = distance from i down to its deepest descendant."""
        dn = self.deepestnode(i)
        return (self.depth(dn) - self.depth(i)) if dn is not None else 0

    def degree(self, i: int) -> int:
        c = self.close(i)
        if c is None:
            return 0
        tgt = self.depth(i) + 1
        return self.count_excess(i+1, c-1, tgt)

    def child(self, i: int, q: int) -> int:
        """
        Opener-index of the qᵗʰ child of opener i (1-based), or None.
        Runs in O(log log n) via a single minselect.
        """
        c = self.close(i)
        if c is None:
            return None

        # children live at excess = depth(i)+1 in (i..c), i.e. [i+1..c-1]
        pos = self.minselect(i+1, c-1, q)
        return pos  # pos will already be an opener or else None

    def childrank(self, i: int) -> int:
        """
        0-based index of i among its siblings—
        i.e. number of siblings strictly to the left.
        """
        p = self.parent(i)
        if p is None:
            return 0
        tgt = self.depth(p) + 1
        # count opens at depth tgt in [p..i-1]
        return self.mincount(p, i-1, tgt)
    
    def set_names(self, names):
        """
        Set the node names array.  Expects an iterable of length n2.
        """
        if len(names) != self.n2:
            raise ValueError("names array must have length n")
        # make a shallow copy so tests can safely reuse their numpy array
        self._names = list(names)

    def set_lengths(self, lengths):
        """
        Set the branch-lengths array.  Expects an iterable of length n2.
        """
        if len(lengths) != self.n2:
            raise ValueError("lengths array must have length n")
        self._lengths = list(lengths)

    def ntips(self) -> int:
        """
        Number of tips (leaves) in the tree.  
        A leaf is an opener immediately followed by a closer.
        """
        count = 0
        for i in range(self.n2 - 1):
            # bitvector: 1=='(', 0==')'
            if self.bv[i] and not self.bv[i+1]:
                count += 1
        return count

    def collapse(self):
        """
        Return a new SuccinctTree in which all unary nodes
        are collapsed (their lengths summed, names inherited).
        """
        # 1. Build an explicit node structure
        class Node:
            def __init__(self, name, length):
                self.name = name
                self.length = length
                self.children = []
        
        stack = []
        root = None
        # iterate over bp string
        for idx in range(self.n2):
            if self.bv[idx]:  # '('
                node = Node(self._names[idx], self._lengths[idx])
                if stack:
                    stack[-1].children.append(node)
                else:
                    root = node
                stack.append(node)
            else:  # ')'
                stack.pop()
        
        # 2. Collapse unary chains
        def collapse_node(u):
            # collapse through single-child chains
            while len(u.children) == 1:
                child = u.children[0]
                u.length += child.length
                u.name = child.name
                u.children = child.children
            # recurse
            for w in u.children:
                collapse_node(w)
        collapse_node(root)
        
        # 3. Reconstruct BP, names, lengths
        bits = []
        names = []
        lengths = []
        def dfs(u):
            bits.append('(')
            names.append(u.name)
            lengths.append(u.length)
            for w in u.children:
                dfs(w)
            bits.append(')')
            names.append(None)
            lengths.append(0.0)
        dfs(root)
        bp_str = ''.join(bits)
        # 4. Build new SuccinctTree and set metadata
        new_tree = SuccinctTree(bp_str)
        new_tree.set_names(names)
        new_tree.set_lengths(lengths)
        return new_tree
    
    def shear(self, in_set: set) -> "SuccinctTree":
        """
        Return a new SuccinctTree containing only the leaves whose names
        are in in_set (and all their ancestors).  Raise ValueError if none.
        """
        # 1) Build a simple Node tree with metadata
        class Node:
            def __init__(self, name, length):
                self.name = name
                self.length = length
                self.children = []

        stack = []
        root = None
        for idx in range(self.n2):
            if self.bv[idx]:                  # opener
                u = Node(self._names[idx], self._lengths[idx])
                if stack:
                    stack[-1].children.append(u)
                else:
                    root = u
                stack.append(u)
            else:                             # closer
                stack.pop()

        # 2) Mark and prune: keep node if it or any descendant has name in in_set
        def mark(u):
            keep = (u.name in in_set)
            new_children = []
            for c in u.children:
                if mark(c):
                    keep = True
                    new_children.append(c)
            u.children = new_children
            return keep

        if not mark(root):
            raise ValueError("tree is empty after shear")

        # 3) Reconstruct BP / names / lengths
        bits, names, lengths = [], [], []
        def dfs(u):
            bits.append('(')
            names.append(u.name)
            lengths.append(u.length)
            for c in u.children:
                dfs(c)
            bits.append(')')
            names.append(None)
            lengths.append(0.0)

        dfs(root)
        bp_str = ''.join(bits)

        # 4) Build the succinct structure & copy metadata
        new = SuccinctTree(bp_str)
        new.set_names(names)
        new.set_lengths(lengths)
        return new
