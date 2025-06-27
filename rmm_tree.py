import math
import array

#------------------------------------------------------------------------------
# Low-level block summary (rmM-tree) utilities
#------------------------------------------------------------------------------
# build_lut16.py  (run ONCE when the module is imported)
#------------------------------------------------------------------------------
LUT16_de   = array.array('h',  [0]*65536)   # signed 16-bit
LUT16_min  = array.array('b',  [0]*65536)   # signed  8-bit
LUT16_max  = array.array('b',  [0]*65536)
LUT16_cntm = array.array('H',  [0]*65536)   # unsigned 16-bit (0..16)

for word in range(65536):
    m   =  16   # larger than any positive prefix
    M   = -16   # smaller than any negative prefix
    cnt = 0
    cur = 0
    for bit in range(16):                        # MSB first
        cur += 1 if (word >> (15-bit)) & 1 else -1
        if cur < m:
            m, cnt = cur, 1
        elif cur == m:
            cnt += 1
        if cur > M:
            M = cur
    e = cur
    LUT16_de[word]   = e
    LUT16_min[word]  = m
    LUT16_max[word]  = M
    LUT16_cntm[word] = cnt

def build_summaries(block):
    """
    Compute (e, m, M, n) summary for a block of booleans:
      e = total excess (+1 for True, -1 for False)
      m = minimum prefix excess within the block
      M = maximum prefix excess within the block
      n = count of prefixes attaining m
    """
    length = len(block)
    if length == 0:
        return 0, 0, 0, 0
    e = 0
    m = length
    M = -length
    n = 0
    for bit in block:
        e += 1 if bit else -1
        if e < m:
            m = e
            n = 1
        elif e == m:
            n += 1
        if e > M:
            M = e
    return e, m, M, n

def combine_blocks(b1, b2):
    """
    Merge two (e, m, M, n) summaries for adjacent blocks.
    """
    e1, m1, M1, n1 = b1
    e2, m2, M2, n2 = b2
    combined_e = e1 + e2
    min1 = m1
    min2 = e1 + m2
    combined_m = min(min1, min2)
    if min1 < min2:
        combined_n = n1
    elif min1 > min2:
        combined_n = n2
    else:
        combined_n = n1 + n2
    combined_M = max(M1, e1 + M2)
    return combined_e, combined_m, combined_M, combined_n


#------------------------------------------------------------------------------
# Per-bucket rmM-tree with O(log log n) bucket-local operations
#------------------------------------------------------------------------------
class BucketRMMinMax:
    """
    One bucket of β bits (β = Θ(log³ n)).
    • Divide the bucket into blocks of b bits  (b = log n · log log n ≈ 1024)
    • Store (e, m, M, n) per block        –––  O(β / b) summaries
    • Build a binary rmM-tree on those     –––  O(β / b) nodes
    Nothing per bit is cached; excess/rmq   queries inside the block
    are answered on-the-fly with the 16-bit LUT.
    """
    def __init__(self, bits, block_size: int = 1024):
        # ------------------------------------------------------------
        # 0. raw data
        # ------------------------------------------------------------
        self.bits = bits = list(bits)               # ensure O(1) indexing
        self.n    = len(bits)
        self.b    = block_size                      # block bits
        self.words_per_block = block_size // 16

        # ------------------------------------------------------------
        # 1. per-block summaries  (e, m, M, n)
        # ------------------------------------------------------------
        self.leaf_summaries = [
            build_summaries(bits[i : i + self.b])
            for i in range(0, self.n, self.b)
        ]                                           # length = #blocks

        self.block_e_prefix = [0]
        for e, _, _, _ in self.leaf_summaries:
            self.block_e_prefix.append(self.block_e_prefix[-1] + e)

        # ------------------------------------------------------------
        # 2. rmM-tree over those summaries
        # ------------------------------------------------------------
        L    = len(self.leaf_summaries)
        self.size = size = 1 << math.ceil(math.log2(L or 1))

        neutral = (0, 0, 0, 0)                      # safer identity
        self.tree = tree = [neutral] * (2 * size)

        for i, summ in enumerate(self.leaf_summaries):
            tree[size + i] = summ

        for idx in range(size - 1, 0, -1):
            tree[idx] = combine_blocks(tree[2 * idx], tree[2 * idx + 1])


    def _word(self, word_idx: int) -> int:
        """Return 16-bit integer for word #word_idx (msb-first)."""
        base = word_idx * 16
        w = 0
        for j in range(16):
            if base + j < self.n and self.bits[base + j]:
                w |= 1 << (15 - j)
        return w

    def _excess_local(self, pos: int) -> int:
        """
        Excess of prefix [0 .. pos] inside this bucket.
        """
        if pos < 0:
            return 0

        blk = pos // self.b
        e   = self.block_e_prefix[blk]            # O(1) prefix on blocks

        blk_base  = blk * self.b
        word_idx  = (pos - blk_base) // 16
        first_word_global = blk_base // 16

        # ---- whole 16-bit words before the current one  (≤ 64) -----------
        for w in range(word_idx):
            e += LUT16_de[self._word(first_word_global + w)]

        # ---- trailing ≤ 15 bits in the current word ----------------------
        bit_start = blk_base + word_idx * 16
        for i in range(bit_start, pos + 1):
            e += 1 if self.bits[i] else -1
        
        # word   = self._word(upto_word)
        # k      = (pos & 15) + 1           # 1 ≤ k ≤ 16
        # mask   = word >> (16 - k)         # MSB-first layout
        # e     += LUT16_de[mask]

        return e

    def _scan_bits(self, start, stop, e0, find_min=True, target=None):
        """
        Scan bits[start:stop]  (≤ 15 positions).

        If find_min:
            return (min_val, first_pos, final_e)

        Else (search for target excess):
            return (first_pos or None, final_e)
        """
        e = e0
        best = (1 << 30) if find_min else None
        best_pos = None

        for i in range(start, stop):
            e += 1 if self.bits[i] else -1
            if find_min:
                if e < best:
                    best, best_pos = e, i
            else:
                if e == target and best_pos is None:
                    best_pos = i        # record first match

        return (best, best_pos, e) if find_min else (best_pos, e)   # else None?

    def _combine_range(self, l_blk, r_blk):
        """Combine summaries across blocks [l_blk..r_blk] in O(log(#blocks))."""
        if l_blk > r_blk:
            return 0, math.inf, -math.inf, 0
        l = self.size + l_blk
        r = self.size + r_blk + 1
        summary = (0, math.inf, -math.inf, 0)
        while l < r:
            if l & 1:
                summary = combine_blocks(summary, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                summary = combine_blocks(summary, self.tree[r])
            l //= 2; r //= 2
        return summary

    # ================================================================
    # 1.  Range-minimum query  (left-most minimum)
    # ================================================================
    def rmq_in_bucket(self, l: int, r: int):
        if l > r:
            l, r = r, l
        best_val = 1 << 30
        best_pos = l

        # ------- prefix excess just before l ---------------------------
        cur = self._excess_local(l - 1) if l else 0

        blk_l, blk_r = l // self.b, r // self.b
        off_l, off_r = l % self.b,  r % self.b

        # ------- 1. partial LEFT block (unchanged) ---------------------
        blk_base = blk_l * self.b
        left_stop = blk_base + (off_r + 1 if blk_l == blk_r else self.b)
        for i in range(blk_base + off_l, min(left_stop, self.n)):
            cur += 1 if self.bits[i] else -1
            if cur < best_val:
                best_val, best_pos = cur, i

        # ------- 2. middle blocks via rmM-tree -------------------------
        if blk_r > blk_l + 1:
            tgt_rel = best_val - cur          # how low we still need to beat
            node, nl, nr = 1, 0, self.size - 1

            # restrict traversal to [blk_l+1 .. blk_r-1]
            ql, qr = blk_l + 1, blk_r - 1

            while nl != nr:
                mid = (nl + nr) >> 1
                left  = node << 1
                right = left | 1

                # Does the left child overlap AND contain a value ≤ tgt_rel?
                if ql <= mid and self.tree[left][1] + cur <= best_val:
                    node, nr = left, mid
                else:
                    # skip left subtree
                    if ql <= mid:
                        cur += self.tree[left][0]   # add its total excess
                    node, nl = right, mid + 1

            blk_mid = nl            # nl == nr
            # scan that one block (≤ b bits) to pinpoint first minimum
            blk_base = blk_mid * self.b
            base = cur
            for i in range(blk_base, min(blk_base + self.b, self.n)):
                base += 1 if self.bits[i] else -1
                if base < best_val:
                    best_val, best_pos = base, i
                    break            # first occurrence is enough

        # ------- 3. partial RIGHT block (unchanged) --------------------
        if blk_r != blk_l:
            blk_base = blk_r * self.b
            for i in range(blk_base, blk_base + off_r + 1):
                cur += 1 if self.bits[i] else -1
                if cur < best_val:
                    best_val, best_pos = cur, i

        return best_pos, best_val

    # ================================================================
    # 2. Range-maximum query  (left-most maximum)
    # ================================================================
    def rMq_in_bucket(self, l: int, r: int):
        if l > r:
            l, r = r, l

        # ---- running excess just before l ---------------------------
        cur = self._excess_local(l - 1) if l else 0
        best_val, best_pos = -1 << 30, l

        blk_l, blk_r = l // self.b, r // self.b
        off_l, off_r = l % self.b,  r % self.b

        # ---------- 1. partial LEFT block ---------------------------
        blk_base  = blk_l * self.b
        left_stop = blk_base + (off_r + 1 if blk_l == blk_r else self.b)

        tmp_cur = cur
        for i in range(blk_base + off_l, min(left_stop, self.n)):
            tmp_cur += 1 if self.bits[i] else -1
            if tmp_cur > best_val:
                best_val, best_pos = tmp_cur, i
        cur = tmp_cur

        # ---------- 2. middle blocks via rmM-tree -------------------
        if blk_r > blk_l + 1:
            # absolute maximum over middle blocks
            _, _, M_mid, _ = self._combine_range(blk_l + 1, blk_r - 1)
            if cur + M_mid > best_val:
                # descend to *left-most* block that attains M_mid
                node, nl, nr = 1, 0, self.size - 1
                ql, qr = blk_l + 1, blk_r - 1
                pref_ex = cur                      # excess before current node
                while nl != nr:                    # O(log log n)
                    mid = (nl + nr) >> 1
                    left  = node << 1
                    right = left | 1
                    mL, ML = self.tree[left][1], self.tree[left][2]
                    if ql <= mid and pref_ex + ML == cur + M_mid:
                        node, nr = left, mid
                    else:
                        if ql <= mid:              # skip left subtree
                            pref_ex += self.tree[left][0]
                        node, nl = right, mid + 1
                blk_mid = nl
                base = pref_ex
                blk_base = blk_mid * self.b
                for i in range(blk_base, min(blk_base + self.b, self.n)):
                    base += 1 if self.bits[i] else -1
                    if base == cur + M_mid:
                        best_pos, best_val = i, base
                        break

        # ---------- 3. partial RIGHT block --------------------------
        if blk_r != blk_l:
            blk_base = blk_r * self.b
            base = cur + self._combine_range(blk_l + 1, blk_r - 1)[0] \
                if blk_r > blk_l + 1 else cur
            for i in range(blk_base, blk_base + off_r + 1):
                base += 1 if self.bits[i] else -1
                if base > best_val:
                    best_val, best_pos = base, i

        return best_pos, best_val

    def _find_first(self, node, nl, nr, ql, qr, target):
        if nr < ql or nl > qr:
            return None
        _, m, M, _ = self.tree[node]
        if m > target or M < target:
            return None
        if nl == nr:
            return nl if nl < len(self.leaf_summaries) else None
        mid = (nl + nr) // 2
        left = self._find_first(2*node,   nl,    mid, ql, qr, target)
        if left is not None:
            return left
        return self._find_first(2*node+1, mid+1, nr, ql, qr, target)

    def _find_last(self, node, nl, nr, ql, qr, target):
        if nr < ql or nl > qr:
            return None
        _, m, M, _ = self.tree[node]
        if m > target or M < target:
            return None
        if nl == nr:
            return nl if nl < len(self.leaf_summaries) else None
        mid = (nl + nr) // 2
        right = self._find_last(2*node+1, mid+1, nr, ql, qr, target)
        if right is not None:
            return right
        return self._find_last(2*node, nl, mid, ql, qr, target)

    def _find_first_block(self, l_blk, r_blk, target):
        return self._find_first(1, 0, self.size - 1, l_blk, r_blk, target)

    def _find_last_block(self, l_blk, r_blk, target):
        return self._find_last(1, 0, self.size - 1, l_blk, r_blk, target)

    # ---------------------------------------------------------------
    # Find first / last offset in THIS bucket whose excess == target
    # ---------------------------------------------------------------
    def find_first_excess(self, target: int):
        blk = self._find_first_block(0,
                                    len(self.leaf_summaries) - 1,
                                    target)
        if blk is None:
            return None

        # O(1) instead of Θ(#blocks)
        cur = self.block_e_prefix[blk]

        start = blk * self.b
        end   = min(start + self.b, self.n)
        for i in range(start, end):
            cur += 1 if self.bits[i] else -1
            if cur == target:
                return i
        return None          # should not happen


    def find_last_excess(self, target: int):
        blk = self._find_last_block(0,
                                    len(self.leaf_summaries) - 1,
                                    target)
        if blk is None:
            return None

        # O(1) prefix excess
        cur = self.block_e_prefix[blk]

        start = blk * self.b
        end   = min(start + self.b, self.n)
        # advance through entire block once
        for i in range(start, end):
            cur += 1 if self.bits[i] else -1

        # scan backward to locate the last occurrence
        for i in range(end - 1, start - 1, -1):
            if cur == target:
                return i
            cur -= 1 if self.bits[i] else -1
        return None          # should not happen

    # ================================================================
    # 3.  Forward search: first j > start with excess == target
    # ================================================================
    def fwdsearch_in_bucket(self, start: int, target: int):
        cur = self._excess_local(start)
        i   = start + 1

        # ---------------- finish current block ----------------------
        blk_end = ((start // self.b) + 1) * self.b
        while i < min(blk_end, self.n):
            cur += 1 if self.bits[i] else -1
            if cur == target:
                return i
            i += 1

        # ----------- descend rmM-tree to find first good block ------
        blk_tot   = len(self.leaf_summaries)
        if i >= self.n:
            return None                        # end of bucket

        # start searching at the first *whole* block after start
        blk      = blk_end // self.b
        node, nl, nr = 1, 0, self.size - 1

        # advance cur by skipping whole subtrees whose [m,M] miss target
        while nl != nr:
            mid  = (nl + nr) >> 1
            left = self.tree[node << 1]
            # Range of left child in absolute terms
            left_m = cur + left[1]
            left_M = cur + left[2]

            if blk <= mid and left_m <= target <= left_M:
                # descend into left child
                node, nr = node << 1, mid
            else:
                # skip whole left subtree
                cur += left[0]
                node, nl = node << 1 | 1, mid + 1

        # nl == nr == block index that must contain the answer
        blk_idx = nl
        base    = cur
        bit     = blk_idx * self.b
        end     = min(bit + self.b, self.n)
        while bit < end:
            base += 1 if self.bits[bit] else -1
            if base == target:
                return bit
            bit += 1
        return None

    # ================================================================
    # 4.  Backward search: last j < start with excess == target
    # ================================================================
    def bwdsearch_in_bucket(self, start: int, target: int):
        """
        Last j < start such that  excess(j) == target,   restricted to this bucket.
        Worst-case O(log log n):
        – ≤ b   bits inside the start block
        – ≤ log₂(#blocks) nodes in the rmM-tree
        – ≤ b   bits in the final block
        """
        if start <= 0:
            return None                         # nothing before pos-0

        # ------------------------------------------------------------
        # 1)  scan backward inside the start block
        # ------------------------------------------------------------
        cur = self._excess_local(start)         # excess at position start
        i   = start
        blk_start = (start // self.b) * self.b
        while i >= blk_start:
            if i < start and cur == target:
                return i
            cur -= 1 if self.bits[i] else -1
            i  -= 1

        # ------------------------------------------------------------
        # 2)  descend the rmM-tree to the last block whose [m,M]
        #     straddles 'target'
        # ------------------------------------------------------------
        if blk_start == 0:
            return None                         # no earlier blocks

        blk = blk_start // self.b - 1           # last *complete* block < start
        node, nl, nr = 1, 0, self.size - 1      # rmM-tree root

        while nl != nr:                         # O(log log n) levels
            mid  = (nl + nr) >> 1
            left = self.tree[node << 1]         # (e, m, M, n) of left child

            if blk > mid:                       # target block is in the *right* child
                cur -= left[0]                  # skip entire left subtree
                node, nl = (node << 1) | 1, mid + 1   # go right
            else:                               # target block is in the *left* child
                node, nr = node << 1, mid              # go left
            # invariant: 'cur' is excess just before subtree 'node'

        blk_idx = nl                            # leaf index that contains answer
        e_blk   = self.leaf_summaries[blk_idx][0]
        excess_at_bit = cur + e_blk             # excess *after* last bit in block

        # ------------------------------------------------------------
        # 3)  scan that block right-to-left
        # ------------------------------------------------------------
        bit = min((blk_idx + 1) * self.b, self.n) - 1
        while bit >= blk_idx * self.b:
            if excess_at_bit == target:
                return bit
            excess_at_bit -= 1 if self.bits[bit] else -1
            bit -= 1

        return None                             # should never fall through

    # ================================================================
    # 5.  mincount: # occurrences of `target` in [l..r]
    # ================================================================
    def mincount_in_bucket(self, l: int, r: int, target: int):
        if l > r:
            l, r = r, l
        cnt = 0
        cur = self._excess_local(l - 1) if l else 0

        blk_l, blk_r = l // self.b, r // self.b
        off_l, off_r = l % self.b, r % self.b

        # ---------- partial LEFT block ---------------------------------
        blk_base   = blk_l * self.b
        left_stop  = blk_base + (off_r + 1 if blk_l == blk_r else self.b)
        for i in range(blk_base + off_l, min(left_stop, self.n)):
            cur += 1 if self.bits[i] else -1
            if cur == target:
                cnt += 1

        # ---------- full middle blocks ---------------------------------
        for blk in range(blk_l + 1, blk_r):
            e_blk, m_blk, _, n_blk = self.leaf_summaries[blk]
            if cur + m_blk == target:
                cnt += n_blk
            cur += e_blk

        # ---------- partial RIGHT block --------------------------------
        if blk_r != blk_l:                     # only when range spans 2+ blocks
            blk_base = blk_r * self.b
            for i in range(blk_base, blk_base + off_r + 1):
                cur += 1 if self.bits[i] else -1
                if cur == target:
                    cnt += 1
        return cnt

    # ================================================================
    # 6.  minselect: position of q-th occurrence of `target` in [l..r]
    # ================================================================
    def minselect_in_bucket(self, l: int, r: int, target: int, q: int):
        if l > r:
            l, r = r, l
        cur = self._excess_local(l - 1) if l else 0

        blk_l, blk_r = l // self.b, r // self.b
        off_l, off_r = l % self.b, r % self.b

        # --- partial left block
        blk_base = blk_l * self.b
        left_stop = blk_base + (off_r + 1 if blk_l == blk_r else self.b)
        for i in range(blk_base + off_l, min(left_stop, self.n)):
            cur += 1 if self.bits[i] else -1
            if cur == target:
                q -= 1
                if q == 0:
                    return i

        # --- full blocks
        for blk in range(blk_l + 1, blk_r):
            e_blk, m_blk, _, n_blk = self.leaf_summaries[blk]
            if cur + m_blk == target:
                if q <= n_blk:
                    # brute scan this block
                    j = blk * self.b
                    cur_blk = cur
                    while j < min((blk + 1) * self.b, self.n):
                        cur_blk += 1 if self.bits[j] else -1
                        if cur_blk == target:
                            q -= 1
                            if q == 0:
                                return j
                        j += 1
                    return None           # shouldn’t happen
                q -= n_blk
            cur += e_blk

        # --- partial right block
        if blk_r != blk_l:
            blk_base = blk_r * self.b
            for i in range(blk_base, blk_base + off_r + 1):
                cur += 1 if self.bits[i] else -1
                if cur == target:
                    q -= 1
                    if q == 0:
                        return i
        return None
