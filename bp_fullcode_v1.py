import math
import unittest
from bitarray import bitarray

class BitVector:
    def __init__(self, bitstring):
        # Convert the input bitstring 
        self.B = bitarray(bitstring)
        self.n = len(self.B)  # Total number of bits in the bitvector
        self.lg_n = int(math.log2(self.n)) if self.n > 0 else 1
        
        # Define block sizes for rank/select structures
        self.superblock_size = 512  # Size of a superblock
        self.subblock_size = 16     # Size of a subblock within a superblock
        # Dynamic block sizes?

        # Initialize structures for rank1 (number of 1s up to a position)
        self.rank1_superblock = []  # Cumulative counts at superblock boundaries
        self.rank1_subblock = []    # Cumulative counts at subblock boundaries within superblocks

        # Initialize structures for rank10 (number of '10' patterns up to a position)
        self.rank10_superblock = []  # Cumulative counts at superblock boundaries
        self.rank10_subblock = []    # Cumulative counts at subblock boundaries within superblocks

        # Precompute structures for rank and select operations
        self.precompute_rank_structures()

    def precompute_rank_structures(self):
        total_ones = 0
        total_ten_patterns = 0

        self.rank1_superblock = []
        self.rank10_superblock = []
        self.rank1_subblock = []
        self.rank10_subblock = []

        subblock_ones = 0
        subblock_ten_patterns = 0

        last_subblock_appended = False

        for i in range(self.n):
            # At superblock boundaries
            if i % self.superblock_size == 0:
                self.rank1_superblock.append(total_ones)
                self.rank10_superblock.append(total_ten_patterns)

            # # At subblock boundaries
            # if i % self.subblock_size == 0 and i != 0:
            #     self.rank1_subblock.append(subblock_ones)
            #     self.rank10_subblock.append(subblock_ten_patterns)
            #     subblock_ones = 0
            #     subblock_ten_patterns = 0

            # Subblock boundary (within superblock)
            if i % self.subblock_size == 0 and i != 0:
                self.rank1_subblock.append(subblock_ones)
                self.rank10_subblock.append(subblock_ten_patterns)
                subblock_ones = 0
                subblock_ten_patterns = 0
                last_subblock_appended = True
            else:
                last_subblock_appended = False

            bit = self.B[i]
            total_ones += bit
            subblock_ones += bit

            if i > 0 and self.B[i - 1] == 1 and self.B[i] == 0:
                total_ten_patterns += 1
                subblock_ten_patterns += 1

        # # Append counts for the last subblock
        # self.rank1_subblock.append(subblock_ones)
        # self.rank10_subblock.append(subblock_ten_patterns)

        # Append the last subblock counts if not already appended
        if not last_subblock_appended:
            self.rank1_subblock.append(subblock_ones)
            self.rank10_subblock.append(subblock_ten_patterns)

        # Store total counts
        # Used to check if select exceeds this
        self.total_ones = total_ones
        self.total_ten_patterns = total_ten_patterns

    def rank1(self, i):
        if i < 0:
            return 0
        # TODO: What return behavior is wanted?
        if i >= self.n:
            return self.total_ones

        superblock_idx = i // self.superblock_size
        subblock_idx = i // self.subblock_size

        # Start with the count from the superblock
        rank = self.rank1_superblock[superblock_idx]

        # Add counts from subblocks within the superblock
        subblock_start_idx = (superblock_idx * self.superblock_size) // self.subblock_size
        rank += sum(self.rank1_subblock[subblock_start_idx:subblock_idx])

        # Scan the remaining bits within the subblock
        start_idx = subblock_idx * self.subblock_size
        for idx in range(start_idx, i + 1):
            rank += self.B[idx]

        return rank

    def rank10(self, i):
        if i < 1:
            return 0
        if i >= self.n:
            return self.total_ten_patterns

        superblock_idx = i // self.superblock_size
        subblock_idx = i // self.subblock_size

        # Start with the count from the superblock
        rank = self.rank10_superblock[superblock_idx]

        # Add counts from subblocks within the superblock
        subblock_start_idx = (superblock_idx * self.superblock_size) // self.subblock_size
        rank += sum(self.rank10_subblock[subblock_start_idx:subblock_idx])

        # Scan the remaining bits within the subblock
        start_idx = max(subblock_idx * self.subblock_size, 1)
        for idx in range(start_idx, i + 1):
            if self.B[idx - 1] == 1 and self.B[idx] == 0:
                rank += 1
        return rank

    def select1(self, k):
        if k <= 0 or k > self.total_ones:
            raise ValueError("k is out of bounds")

        # Step 1: Binary search over superblocks
        left, right = 0, len(self.rank1_superblock) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.rank1_superblock[mid] < k:
                left = mid + 1
            else:
                right = mid - 1
        superblock_idx = right

        # Adjust k to be relative to the superblock
        k -= self.rank1_superblock[superblock_idx]

        # Step 2: Linear search over subblocks within the superblock
        superblock_start = superblock_idx * self.superblock_size
        num_subblocks = (min(self.n - superblock_start, self.superblock_size) + self.subblock_size - 1) // self.subblock_size
        subblock_start_idx = (superblock_start) // self.subblock_size

        for i in range(num_subblocks):
            subblock_idx = subblock_start_idx + i
            ones_in_subblock = self.rank1_subblock[subblock_idx]
            if ones_in_subblock >= k:
                # Step 3: Scan within the subblock
                start_idx = superblock_start + i * self.subblock_size
                end_idx = min(start_idx + self.subblock_size, self.n)
                count = 0
                for idx in range(start_idx, end_idx):
                    if self.B[idx] == 1:
                        count += 1
                        if count == k:
                            return idx
                raise ValueError("k is out of bounds within subblock")
            else:
                k -= ones_in_subblock

        # If not found in subblocks, scan remaining bits in the superblock
        start_idx = superblock_start + num_subblocks * self.subblock_size
        end_idx = min(superblock_start + self.superblock_size, self.n)
        count = 0
        for idx in range(start_idx, end_idx):
            if self.B[idx] == 1:
                count += 1
                if count == k:
                    return idx

        raise ValueError("k is out of bounds after scanning subblocks")

    def select10(self, k):
        if k <= 0 or k > self.total_ten_patterns:
            raise ValueError("k is out of bounds")

        # Step 1: Binary search over superblocks
        left, right = 0, len(self.rank10_superblock) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.rank10_superblock[mid] < k:
                left = mid + 1
            else:
                right = mid - 1
        superblock_idx = right

        # Adjust k to be relative to the superblock
        k -= self.rank10_superblock[superblock_idx]

        # Step 2: Linear search over subblocks within the superblock
        superblock_start = superblock_idx * self.superblock_size
        num_subblocks = (min(self.n - superblock_start, self.superblock_size) + self.subblock_size - 1) // self.subblock_size
        subblock_start_idx = (superblock_start) // self.subblock_size

        for i in range(num_subblocks):
            subblock_idx = subblock_start_idx + i
            tens_in_subblock = self.rank10_subblock[subblock_idx]
            if tens_in_subblock >= k:
                # Step 3: Scan within the subblock
                start_idx = max(superblock_start + i * self.subblock_size, 1)
                end_idx = min(start_idx + self.subblock_size, self.n)
                count = 0
                for idx in range(start_idx, end_idx):
                    if self.B[idx - 1] == 1 and self.B[idx] == 0:
                        count += 1
                        if count == k:
                            return idx - 1
                raise ValueError("k is out of bounds within subblock")
            else:
                k -= tens_in_subblock

        # If not found in subblocks, scan remaining bits in the superblock
        start_idx = superblock_start + num_subblocks * self.subblock_size
        end_idx = min(superblock_start + self.superblock_size, self.n)
        count = 0
        for idx in range(max(start_idx, 1), end_idx):
            if self.B[idx - 1] == 1 and self.B[idx] == 0:
                count += 1
                if count == k:
                    return idx - 1

        raise ValueError("k is out of bounds after scanning subblocks")
    

class RMMinMaxTree:
    def __init__(self, bp_tree):
        # Initialize the RMMinMaxTree with a given balanced parentheses tree (bp_tree).
        self.bp_tree = bp_tree  # Assume bp_tree has a 'bitvector' attribute of type BitVector.
        self.n = self.bp_tree.bitvector.n  # Total number of bits in the BP sequence.

        # Define the block size for dividing the sequence; adjust as needed.
        # The block size is set to half the logarithm base 2 of n.
        self.block_size = max(1, int(0.5 * math.log2(self.n)))  # Adjust as needed.
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size  # Total number of blocks.

        # Initialize block-level data structures.
        self.block_mins = []            # Minimum excess value in each block.
        self.block_maxs = []            # Maximum excess value in each block.
        self.block_min_positions = []   # Position of the minimum excess within each block.
        self.block_min_counts = []      # Number of times the minimum excess occurs in each block.
        self.block_max_positions = []   # Position of the maximum excess within each block.
        self.block_max_counts = []      # Number of times the maximum excess occurs in each block.
        self.block_excess_starts = []   # Excess at the start of each block.
        self.block_excess_ends = []     # Excess at the end of each block.

        # Build the necessary structures for efficient queries.
        self._build_blocks()            # Preprocess blocks to compute minima, maxima, and excesses.
        self._build_sparse_tables()     # Build sparse tables over block minima and maxima for RMQ queries.
        self._build_inblock_structures()  # Build in-block RMQ structures, sharing among identical blocks.

    def _build_blocks(self):
        """Preprocess blocks to compute minima, maxima, and excess values at block boundaries."""
        for block_idx in range(self.num_blocks):
            # Calculate the start and end indices for the block.
            start = block_idx * self.block_size
            end = min((block_idx + 1) * self.block_size, self.n) - 1

            # Compute excess at block boundaries.
            if start == 0:
                excess_start = 0  # Excess at the start of the first block is 0.
            else:
                excess_start = self.excess(start - 1)  # Excess at the position before the block.
            excess_end = self.excess(end)  # Excess at the end of the block.

            # Store the block's starting and ending excesses.
            self.block_excess_starts.append(excess_start)
            self.block_excess_ends.append(excess_end)

            # Compute block minima and maxima.
            min_value, min_pos, min_count, max_value, max_pos, max_count = self._compute_block_min_max(block_idx, excess_start)
            self.block_mins.append(min_value)                # Minimum excess value in the block.
            self.block_min_positions.append(min_pos)         # Position of minimum excess within the block.
            self.block_min_counts.append(min_count)          # Count of minimum excess occurrences.
            self.block_maxs.append(max_value)                # Maximum excess value in the block.
            self.block_max_positions.append(max_pos)         # Position of maximum excess within the block.
            self.block_max_counts.append(max_count)          # Count of maximum excess occurrences.

    def _compute_block_min_max(self, block_idx, excess_start):
        """Compute the minimum and maximum excess values within a block."""
        start = block_idx * self.block_size
        end = min((block_idx + 1) * self.block_size, self.n)
        min_value = float('inf')        # Initialize minimum excess value.
        max_value = float('-inf')       # Initialize maximum excess value.
        min_positions = []              # Positions where the minimum excess occurs.
        max_positions = []              # Positions where the maximum excess occurs.
        current_excess = excess_start   # Initialize current excess at block start.
        min_count = 0                   # Initialize count of minimum excess occurrences.
        max_count = 0                   # Initialize count of maximum excess occurrences.

        for idx in range(start, end):
            # Read the bit at position idx; 1 for '(', 0 for ')'.
            bit = self.bp_tree.bitvector.B[idx]
            # Update the current excess: increment for '(', decrement for ')'.
            current_excess += 1 if bit else -1

            # Update minimum excess if necessary.
            if current_excess < min_value:
                min_value = current_excess
                min_positions = [idx]
                min_count = 1
            elif current_excess == min_value:
                min_positions.append(idx)
                min_count += 1

            # Update maximum excess if necessary.
            if current_excess > max_value:
                max_value = current_excess
                max_positions = [idx]
                max_count = 1
            elif current_excess == max_value:
                max_positions.append(idx)
                max_count += 1

        # Choose the first occurrence of the minimum and maximum excess.
        min_pos = min_positions[0] if min_positions else start
        max_pos = max_positions[0] if max_positions else start
        return min_value, min_pos, min_count, max_value, max_pos, max_count

    def _build_sparse_tables(self):
        """Build sparse tables over block minima and maxima for RMQ (Range Minimum/Maximum Query) queries."""
        k = int(math.log2(self.num_blocks)) + 1  # Maximum power of 2 needed.

        # Build sparse table for minima.
        self.sparse_table_min = [self.block_mins.copy()]  # First row is the block minima.

        # Build the sparse table by computing minimums over ranges of size 2^j.
        for j in range(1, k):
            prev_row = self.sparse_table_min[-1]  # Previous row in the sparse table.
            curr_row = []                         # Current row to be computed.
            for i in range(len(prev_row) - (1 << (j - 1))):
                # Compare two overlapping intervals of size 2^(j-1).
                if prev_row[i] <= prev_row[i + (1 << (j - 1))]:
                    curr_row.append(prev_row[i])
                else:
                    curr_row.append(prev_row[i + (1 << (j - 1))])
            self.sparse_table_min.append(curr_row)  # Add the current row to the sparse table.

        # Build sparse table for maxima.
        self.sparse_table_max = [self.block_maxs.copy()]  # First row is the block maxima.

        # Build the sparse table by computing maximums over ranges of size 2^j.
        for j in range(1, k):
            prev_row = self.sparse_table_max[-1]  # Previous row in the sparse table.
            curr_row = []                         # Current row to be computed.
            for i in range(len(prev_row) - (1 << (j - 1))):
                # Compare two overlapping intervals of size 2^(j-1).
                if prev_row[i] >= prev_row[i + (1 << (j - 1))]:
                    curr_row.append(prev_row[i])
                else:
                    curr_row.append(prev_row[i + (1 << (j - 1))])
            self.sparse_table_max.append(curr_row)  # Add the current row to the sparse table.

    def _build_inblock_structures(self):
        """Build in-block RMQ structures, sharing them among identical blocks."""
        self.block_types = []             # List of block types (normalized excess sequences).
        self.block_type_structures = {}   # Mapping from block type to its in-block RMQ structures.

        for block_idx in range(self.num_blocks):
            # Compute the excess sequence within the block.
            block_excess = self._compute_block_excess(block_idx)
            # Normalize the excess sequence by subtracting the first excess value.
            normalized_excess = tuple(val - block_excess[0] for val in block_excess)
            block_type = normalized_excess  # Use the normalized excess sequence as block type.
            self.block_types.append(block_type)

            if block_type not in self.block_type_structures:
                # Build in-block RMQ structures (sparse tables) for this block type.
                min_st, max_st = self._build_inblock_rmq(block_excess)
                self.block_type_structures[block_type] = {'min': min_st, 'max': max_st}

    def _compute_block_excess(self, block_idx):
        """Compute the excess values within a block."""
        start = block_idx * self.block_size
        end = min((block_idx + 1) * self.block_size, self.n)
        excess_start = self.block_excess_starts[block_idx]  # Excess at the start of the block.
        block_excess = []            # List to store excess values within the block.
        current_excess = excess_start

        for idx in range(start, end):
            bit = self.bp_tree.bitvector.B[idx]
            current_excess += 1 if bit else -1
            block_excess.append(current_excess)

        return block_excess

    def _build_inblock_rmq(self, block_excess):
        """Build RMQ structures for a block using sparse tables for both min and max."""
        n = len(block_excess)                      # Size of the block.
        k = int(math.log2(n)) + 1                  # Maximum power of 2 needed.

        # Build sparse table for minima.
        st_min = [block_excess.copy()]             # First row is the block excess sequence.

        for j in range(1, k):
            prev_row = st_min[-1]                  # Previous row in the sparse table.
            curr_row = []                          # Current row to be computed.
            for i in range(len(prev_row) - (1 << (j - 1))):
                if prev_row[i] <= prev_row[i + (1 << (j - 1))]:
                    curr_row.append(prev_row[i])
                else:
                    curr_row.append(prev_row[i + (1 << (j - 1))])
            st_min.append(curr_row)                # Add the current row to the sparse table.

        # Build sparse table for maxima.
        st_max = [block_excess.copy()]             # First row is the block excess sequence.

        for j in range(1, k):
            prev_row = st_max[-1]                  # Previous row in the sparse table.
            curr_row = []                          # Current row to be computed.
            for i in range(len(prev_row) - (1 << (j - 1))):
                if prev_row[i] >= prev_row[i + (1 << (j - 1))]:
                    curr_row.append(prev_row[i])
                else:
                    curr_row.append(prev_row[i + (1 << (j - 1))])
            st_max.append(curr_row)                # Add the current row to the sparse table.

        return st_min, st_max

    def excess(self, i):
        """Compute the excess at position i on-the-fly."""
        if i < 0 or i >= self.n:
            raise IndexError("Index out of bounds")
        rank1_i = self.bp_tree.bitvector.rank1(i)  # Number of '1's up to position i.
        return 2 * rank1_i - i                     # Excess is 2*rank1(i) - i.

    def depth(self, i):
        """Return the depth of the node at position i."""
        return self.excess(i)                      # Depth is equivalent to excess at position i.

    def rmq(self, i, j):
        """Find the position of the minimum excess between positions i and j."""
        if i > j:
            i, j = j, i  # Swap i and j if they are in the wrong order.

        if i < 0 or j >= self.n:
            raise IndexError("Index out of bounds")

        block_i = i // self.block_size  # Block index for position i.
        block_j = j // self.block_size  # Block index for position j.

        if block_i == block_j:
            # If i and j are within the same block, perform in-block RMQ.
            idx = self._inblock_rmq(block_i, i % self.block_size, j % self.block_size)
            return block_i * self.block_size + idx
        else:
            # Left partial block (from i to end of block_i).
            left_idx, left_min = self._inblock_rmq_with_excess(block_i, i % self.block_size, self.block_size - 1)
            left_pos = block_i * self.block_size + left_idx
            left_min += self.block_excess_starts[block_i]  # Adjust min value with block start excess.

            # Right partial block (from start of block_j to j).
            right_idx, right_min = self._inblock_rmq_with_excess(block_j, 0, j % self.block_size)
            right_pos = block_j * self.block_size + right_idx
            right_min += self.block_excess_starts[block_j]

            # Middle blocks (from block_i + 1 to block_j - 1).
            if block_j - block_i > 1:
                # Use the sparse table to find the minimum among the middle blocks.
                k = int(math.log2(block_j - block_i - 1))
                st = self.sparse_table_min
                mid_block_start = block_i + 1
                mid_block_end = block_j - 1 - (1 << k) + 1
                left_interval_min = st[k][mid_block_start]
                right_interval_min = st[k][mid_block_end]

                if left_interval_min <= right_interval_min:
                    mid_min = left_interval_min
                    mid_pos = self.block_min_positions[mid_block_start]
                else:
                    mid_min = right_interval_min
                    mid_pos = self.block_min_positions[mid_block_end]
            else:
                # No middle blocks.
                mid_min = float('inf')
                mid_pos = -1

            # Adjust mid_min with block excess.
            if mid_min != float('inf'):
                mid_min += self.block_excess_starts[block_i + 1]

            # Compare mins from left, middle, and right to find the overall minimum.
            min_value = min(left_min, mid_min, right_min)
            if min_value == left_min:
                return left_pos
            elif min_value == mid_min:
                return mid_pos
            else:
                return right_pos

    def rmq_max(self, i, j):
        """Find the position of the maximum excess between positions i and j."""
        if i > j:
            i, j = j, i  # Swap i and j if they are in the wrong order.

        if i < 0 or j >= self.n:
            raise IndexError("Index out of bounds")

        block_i = i // self.block_size  # Block index for position i.
        block_j = j // self.block_size  # Block index for position j.

        if block_i == block_j:
            # If i and j are within the same block, perform in-block RMQ for maxima.
            idx = self._inblock_rmq_max(block_i, i % self.block_size, j % self.block_size)
            return block_i * self.block_size + idx
        else:
            # Left partial block (from i to end of block_i).
            left_idx, left_max = self._inblock_rmq_with_excess_max(block_i, i % self.block_size, self.block_size - 1)
            left_pos = block_i * self.block_size + left_idx
            left_max += self.block_excess_starts[block_i]  # Adjust max value with block start excess.

            # Right partial block (from start of block_j to j).
            right_idx, right_max = self._inblock_rmq_with_excess_max(block_j, 0, j % self.block_size)
            right_pos = block_j * self.block_size + right_idx
            right_max += self.block_excess_starts[block_j]

            # Middle blocks (from block_i + 1 to block_j - 1).
            if block_j - block_i > 1:
                # Use the sparse table to find the maximum among the middle blocks.
                k = int(math.log2(block_j - block_i - 1))
                st = self.sparse_table_max
                mid_block_start = block_i + 1
                mid_block_end = block_j - 1 - (1 << k) + 1
                left_interval_max = st[k][mid_block_start]
                right_interval_max = st[k][mid_block_end]

                if left_interval_max >= right_interval_max:
                    mid_max = left_interval_max
                    mid_pos = self.block_max_positions[mid_block_start]
                else:
                    mid_max = right_interval_max
                    mid_pos = self.block_max_positions[mid_block_end]
            else:
                # No middle blocks.
                mid_max = float('-inf')
                mid_pos = -1

            # Adjust mid_max with block excess.
            if mid_max != float('-inf'):
                mid_max += self.block_excess_starts[block_i + 1]

            # Compare maxes from left, middle, and right to find the overall maximum.
            max_value = max(left_max, mid_max, right_max)
            if max_value == left_max:
                return left_pos
            elif max_value == mid_max:
                return mid_pos
            else:
                return right_pos

    def _inblock_rmq(self, block_index, l, r):
        """Perform RMQ within a block using precomputed in-block structures (minima)."""
        block_type = self.block_types[block_index]                # Get the block type (normalized excess sequence).
        st = self.block_type_structures[block_type]['min']        # Get the in-block RMQ sparse table for minima.
        j = int(math.log2(r - l + 1))                             # Compute the largest power of 2 <= range size.
        left = st[j][l]                                           # Minimum in the interval starting at l.
        right = st[j][r - (1 << j) + 1]                           # Minimum in the interval ending at r.

        if left <= right:
            return l
        else:
            return r - (1 << j) + 1

    def _inblock_rmq_with_excess(self, block_index, l, r):
        """Perform RMQ within a block and return both index and min value."""
        block_type = self.block_types[block_index]  # Get the block type.
        st = self.block_type_structures[block_type]['min']  # Get the in-block RMQ sparse table for minima.
        j = int(math.log2(r - l + 1))               # Compute the largest power of 2 <= range size.
        left_val = st[j][l]                         # Minimum value in the interval starting at l.
        right_val = st[j][r - (1 << j) + 1]         # Minimum value in the interval ending at r.

        if left_val <= right_val:
            idx = l
            min_val = left_val
        else:
            idx = r - (1 << j) + 1
            min_val = right_val

        return idx, min_val

    def _inblock_rmq_max(self, block_index, l, r):
        """Perform RMQ within a block using precomputed in-block structures (maxima)."""
        block_type = self.block_types[block_index]                # Get the block type (normalized excess sequence).
        st = self.block_type_structures[block_type]['max']        # Get the in-block RMQ sparse table for maxima.
        j = int(math.log2(r - l + 1))                             # Compute the largest power of 2 <= range size.
        left = st[j][l]                                           # Maximum in the interval starting at l.
        right = st[j][r - (1 << j) + 1]                           # Maximum in the interval ending at r.

        if left >= right:
            return l
        else:
            return r - (1 << j) + 1

    def _inblock_rmq_with_excess_max(self, block_index, l, r):
        """Perform RMQ within a block and return both index and max value."""
        block_type = self.block_types[block_index]  # Get the block type.
        st = self.block_type_structures[block_type]['max']  # Get the in-block RMQ sparse table for maxima.
        j = int(math.log2(r - l + 1))               # Compute the largest power of 2 <= range size.
        left_val = st[j][l]                         # Maximum value in the interval starting at l.
        right_val = st[j][r - (1 << j) + 1]         # Maximum value in the interval ending at r.

        if left_val >= right_val:
            idx = l
            max_val = left_val
        else:
            idx = r - (1 << j) + 1
            max_val = right_val

        return idx, max_val

    def fwdsearch(self, i, d):
        """Find the smallest j > i such that excess(j) = excess(i) + d."""
        if i < 0 or i >= self.n:
            raise IndexError("Index out of bounds")

        target_excess = self.excess(i) + d  # The target excess we're searching for.
        block_i = i // self.block_size      # Block index for position i.
        current_excess = self.excess(i)     # Current excess at position i.

        # Search within the current block.
        block_end = min((block_i + 1) * self.block_size, self.n)
        for idx in range(i + 1, block_end):
            bit = self.bp_tree.bitvector.B[idx]
            current_excess += 1 if bit else -1
            if current_excess == target_excess:
                return idx

        # Search across blocks.
        for block_idx in range(block_i + 1, self.num_blocks):
            block_start = block_idx * self.block_size
            block_end = min((block_idx + 1) * self.block_size, self.n)
            block_excess_start = self.block_excess_starts[block_idx]
            min_excess_in_block = self.block_mins[block_idx] + block_excess_start
            max_excess_in_block = self.block_maxs[block_idx] + block_excess_start

            if min_excess_in_block <= target_excess <= max_excess_in_block:
                # The target excess could be in this block.
                current_excess = block_excess_start
                for idx in range(block_start, block_end):
                    bit = self.bp_tree.bitvector.B[idx]
                    current_excess += 1 if bit else -1
                    if current_excess == target_excess:
                        return idx
            else:
                continue  # Skip this block.

        return -1  # Target excess not found.

    def bwdsearch(self, i, d):
        """Find the largest j < i such that excess(j) = excess(i) + d."""
        if i < 0 or i >= self.n:
            raise IndexError("Index out of bounds")

        target_excess = self.excess(i) + d  # The target excess we're searching for.
        block_i = i // self.block_size      # Block index for position i.
        current_excess = self.excess(i)     # Current excess at position i.

        # Search within the current block.
        block_start = block_i * self.block_size
        for idx in range(i - 1, block_start - 1, -1):
            bit = self.bp_tree.bitvector.B[idx]
            current_excess -= 1 if bit else -1
            if current_excess == target_excess:
                return idx

        # Search across blocks.
        for block_idx in range(block_i - 1, -1, -1):
            block_start = block_idx * self.block_size
            block_end = min((block_idx + 1) * self.block_size, self.n)
            block_excess_end = self.block_excess_ends[block_idx]
            min_excess_in_block = self.block_mins[block_idx] + self.block_excess_starts[block_idx]
            max_excess_in_block = self.block_maxs[block_idx] + self.block_excess_starts[block_idx]

            if min_excess_in_block <= target_excess <= max_excess_in_block:
                # The target excess could be in this block.
                current_excess = self.block_excess_ends[block_idx]
                for idx in range(block_end - 1, block_start - 1, -1):
                    bit = self.bp_tree.bitvector.B[idx]
                    current_excess -= 1 if bit else -1
                    if current_excess == target_excess:
                        return idx
            else:
                continue  # Skip this block.

        return -1  # Target excess not found.

    def find_close(self, i):
        """Find the matching closing parenthesis for the opening parenthesis at position i."""
        return self.fwdsearch(i, -1)  # Find next position where excess decreases by 1.

    def find_open(self, i):
        """Find the matching opening parenthesis for the closing parenthesis at position i."""
        idx = self.bwdsearch(i, 0)  # Find previous position where excess is the same.
        if idx == -1:
            return -1
        return idx + 1

    def enclose(self, i):
        """Find the opening parenthesis of the smallest matching pair that encloses position i."""
        idx = self.bwdsearch(i, -2)  # Find previous position where excess decreases by 2.
        if idx == -1:
            return -1
        return idx + 1

    def isancestor(self, i, j):
        """Check if node at position i is an ancestor of node at position j."""
        close_i = self.find_close(i)  # Find the closing parenthesis matching i.
        return i <= j < close_i       # j is within the range of i and its matching closing parenthesis.

    def lca(self, i, j):
        """Find the lowest common ancestor (LCA) of nodes at positions i and j."""
        if i > j:
            i, j = j, i  # Swap i and j if they are in the wrong order.

        if self.isancestor(i, j):
            return i  # If i is an ancestor of j, then i is the LCA.
        if self.isancestor(j, i):
            return j  # If j is an ancestor of i, then j is the LCA.

        k = self.rmq(i, j)  # Find position with minimum excess between i and j.
        return self.enclose(k)  # The LCA is the node enclosing the minimum excess position.

    def mincount(self, i, j):
        """Count the number of times the minimum excess occurs between positions i and j."""
        if i > j:
            i, j = j, i

        if i < 0 or j >= self.n:
            raise IndexError("Index out of bounds")

        min_value = self.excess(self.rmq(i, j))  # Minimum excess value in the range.
        count = 0
        block_i = i // self.block_size
        block_j = j // self.block_size

        # Left partial block (from i to end of block_i or j).
        for idx in range(i, min((block_i + 1) * self.block_size, j + 1)):
            if self.excess(idx) == min_value:
                count += 1

        # Middle blocks (from block_i + 1 to block_j - 1).
        for blk in range(block_i + 1, block_j):
            block_min = self.block_mins[blk] + self.block_excess_starts[blk]
            if block_min == min_value:
                count += self.block_min_counts[blk]

        # Right partial block (from start of block_j to j).
        if block_j != block_i:
            for idx in range(block_j * self.block_size, j + 1):
                if self.excess(idx) == min_value:
                    count += 1

        return count

    def minselect(self, i, j, q):
        """Find the position of the q-th occurrence of the minimum excess between positions i and j."""
        if i > j:
            i, j = j, i

        if i < 0 or j >= self.n:
            raise IndexError("Index out of bounds")

        min_value = self.excess(self.rmq(i, j))  # Minimum excess value in the range.
        count = 0
        block_i = i // self.block_size
        block_j = j // self.block_size

        # Left partial block.
        for idx in range(i, min((block_i + 1) * self.block_size, j + 1)):
            if self.excess(idx) == min_value:
                count += 1
                if count == q:
                    return idx

        # Middle blocks.
        for blk in range(block_i + 1, block_j):
            block_min = self.block_mins[blk] + self.block_excess_starts[blk]
            if block_min == min_value:
                blk_count = self.block_min_counts[blk]
                if count + blk_count >= q:
                    # The q-th occurrence is in this block.
                    # Scan within the block to find the exact position.
                    start_idx = blk * self.block_size
                    end_idx = min((blk + 1) * self.block_size, self.n)
                    for idx in range(start_idx, end_idx):
                        if self.excess(idx) == min_value:
                            count += 1
                            if count == q:
                                return idx
                    raise Exception("Error in minselect")  # Should not reach here.
                else:
                    count += blk_count  # Accumulate count.
        # Right partial block.
        if block_j != block_i:
            for idx in range(block_j * self.block_size, j + 1):
                if self.excess(idx) == min_value:
                    count += 1
                    if count == q:
                        return idx

        raise ValueError("q exceeds the number of minimum occurrences")  # q is too large.


class BP_Tree:
    def __init__(self, bitstring):
        """
        Initializes the BP_Tree with a given bitstring.

        Parameters:
        - bitstring (str): A string consisting of '0's and '1's representing the BP sequence.
        """
        self.bitvector = BitVector(bitstring)
        self.n = self.bitvector.n
        self.rmmtree = RMMinMaxTree(self)  # Pass self to RMMinMaxTree

    def find_close(self, i):
        """Find the matching closing parenthesis for the opening parenthesis at position i."""
        return self.rmmtree.find_close(i)

    def find_open(self, i):
        """Find the matching opening parenthesis for the closing parenthesis at position i."""
        return self.rmmtree.find_open(i)

    def enclose(self, i):
        """Find the smallest matching pair that encloses position i."""
        return self.rmmtree.enclose(i)

    def isancestor(self, i, j):
        """Check if node at position i is an ancestor of node at position j."""
        return self.rmmtree.isancestor(i, j)

    def lca(self, i, j):
        """Find the lowest common ancestor of nodes at positions i and j."""
        return self.rmmtree.lca(i, j)

    def depth(self, i):
        """Return the depth of the node at position i."""
        return self.rmmtree.depth(i)

    def excess(self, i):
        """Compute the excess at position i."""
        return self.rmmtree.excess(i)



##### TESTING #####

class TestBitVector(unittest.TestCase):
    def test_basic_rank1_rank10_operations(self):
        """Test Case 1: Basic rank1 and rank10 operations on '1100101'"""
        print("Test Case 1: Basic rank1 and rank10 operations on '1100101'")
        bv = BitVector("1100101")

        # rank1 tests
        self.assertEqual(bv.rank1(0), 1, "rank1(0)")
        self.assertEqual(bv.rank1(3), 2, "rank1(3)")
        self.assertEqual(bv.rank1(6), 4, "rank1(6)")

        # select1 tests
        self.assertEqual(bv.select1(1), 0, "select1(1)")
        self.assertEqual(bv.select1(2), 1, "select1(2)")
        self.assertEqual(bv.select1(4), 6, "select1(4)")

        # rank10 tests
        self.assertEqual(bv.rank10(0), 0, "rank10(0)")
        self.assertEqual(bv.rank10(4), 1, "rank10(4)")
        self.assertEqual(bv.rank10(6), 2, "rank10(6)")

        # select10 tests
        self.assertEqual(bv.select10(1), 1, "select10(1)")
        self.assertEqual(bv.select10(2), 4, "select10(2)")

    def test_empty_bitvector(self):
        """Test Case 2: Edge Case with empty bitstring"""
        print("Test Case 2: Edge Case with empty bitstring")
        bv_empty = BitVector("")
        self.assertEqual(bv_empty.rank1(0), 0, "rank1 on empty bitvector")
        self.assertEqual(bv_empty.rank10(0), 0, "rank10 on empty bitvector")
        with self.assertRaises(ValueError, msg="select1(1) should raise ValueError for empty bitvector"):
            bv_empty.select1(1)
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for empty bitvector"):
            bv_empty.select10(1)
        print("Empty bitvector operations passed.\n")

    def test_all_ones(self):
        """Test Case 3: BitVector with only '1's"""
        print("Test Case 3: BitVector with only '1's")
        bv_ones = BitVector("111111")
        self.assertEqual(bv_ones.rank1(5), 6, "rank1(5) - All Ones")
        self.assertEqual(bv_ones.select1(4), 3, "select1(4) - All Ones")
        self.assertEqual(bv_ones.rank10(5), 0, "rank10(5) - All Ones")
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for all-ones bitvector"):
            bv_ones.select10(1)
        print("All-ones bitvector operations passed.\n")

    def test_all_zeros(self):
        """Test Case 4: BitVector with only '0's"""
        print("Test Case 4: BitVector with only '0's")
        bv_zeros = BitVector("000000")
        self.assertEqual(bv_zeros.rank1(5), 0, "rank1(5) - All Zeros")
        self.assertEqual(bv_zeros.rank10(5), 0, "rank10(5) - All Zeros")
        with self.assertRaises(ValueError, msg="select1(1) should raise ValueError for all-zeros bitvector"):
            bv_zeros.select1(1)
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for all-zeros bitvector"):
            bv_zeros.select10(1)
        print("All-zeros bitvector operations passed.\n")

    def test_alternating_bits(self):
        """Test Case 5: Complex pattern '101010'"""
        print("Test Case 5: Complex pattern '101010'")
        bv_pattern = BitVector("101010")
        self.assertEqual(bv_pattern.rank1(5), 3, "rank1(5) - Alternating")
        self.assertEqual(bv_pattern.rank10(5), 3, "rank10(5) - Alternating")
        self.assertEqual(bv_pattern.select1(2), 2, "select1(2) - Alternating")
        self.assertEqual(bv_pattern.select10(1), 0, "select10(1) - Alternating")
        self.assertEqual(bv_pattern.select10(2), 2, "select10(2) - Alternating")
        with self.assertRaises(ValueError, msg="select10(4) should raise ValueError for '101010' bitvector"):
            bv_pattern.select10(4)
        print("Complex pattern operations passed.\n")

    def test_single_one(self):
        """Test Case 6: Single '1'"""
        print("Test Case 6: Single '1'")
        bv_single_one = BitVector("1")
        self.assertEqual(bv_single_one.rank1(0), 1, "rank1(0) - Single '1'")
        self.assertEqual(bv_single_one.rank10(0), 0, "rank10(0) - Single '1'")
        self.assertEqual(bv_single_one.select1(1), 0, "select1(1) - Single '1'")
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for single '1' bitvector"):
            bv_single_one.select10(1)
        print("Single '1' bitvector operations passed.\n")

    def test_single_zero(self):
        """Test Case 7: Single '0'"""
        print("Test Case 7: Single '0'")
        bv_single_zero = BitVector("0")
        self.assertEqual(bv_single_zero.rank1(0), 0, "rank1(0) - Single '0'")
        self.assertEqual(bv_single_zero.rank10(0), 0, "rank10(0) - Single '0'")
        with self.assertRaises(ValueError, msg="select1(1) should raise ValueError for single '0' bitvector"):
            bv_single_zero.select1(1)
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for single '0' bitvector"):
            bv_single_zero.select10(1)
        print("Single '0' bitvector operations passed.\n")

    def test_out_of_bounds_rank(self):
        """Test Case 8: Out-of-Bounds Indices for `rank`"""
        print("Test Case 8: Out-of-Bounds Indices for `rank`")
        bv = BitVector("1100101")
        self.assertEqual(bv.rank1(10), bv.total_ones, "rank1(10) - Out of Bounds")
        self.assertEqual(bv.rank10(10), bv.total_ten_patterns, "rank10(10) - Out of Bounds")
        print("Out-of-bounds rank operations passed.\n")

    def test_invalid_select(self):
        """Test Case 9: Invalid Occurrences for `select`"""
        print("Test Case 9: Invalid Occurrences for `select`")
        bv = BitVector("1100101")
        with self.assertRaises(ValueError, msg="select1(0) should raise ValueError"):
            bv.select1(0)
        with self.assertRaises(ValueError, msg="select1(5) should raise ValueError"):
            bv.select1(5)
        with self.assertRaises(ValueError, msg="select10(0) should raise ValueError"):
            bv.select10(0)
        with self.assertRaises(ValueError, msg="select10(3) should raise ValueError"):
            bv.select10(3)
        print("Invalid select operations passed.\n")

    def test_large_bitvector_stress(self):
        """Test Case 10: Large BitVector Stress Test"""
        print("Test Case 10: Large BitVector Stress Test")
        large_bv = BitVector("101" * 1000)  # 3000-bit pattern
        self.assertEqual(large_bv.rank1(2999), 2000, "rank1(2999) - Large BitVector")
        self.assertEqual(large_bv.rank10(2999), 1000, "rank10(2999) - Large BitVector")
        self.assertEqual(large_bv.select1(500), 749, "select1(500) - Large BitVector")
        self.assertEqual(large_bv.select10(500), 1497, "select10(500) - Large BitVector")
        print("Large BitVector operations passed.\n")


class TestRMMinMaxTree(unittest.TestCase):
    def setUp(self):
        # Define BP sequences
        self.bp_sequences = {
            'empty': '',
            'single_node': '()',
            'simple_tree': '(()(()))',
            'left_skewed': '((((()))))',
            'right_skewed': '(()()()())',
            'complex_tree': '(()(()())(()()))',
            'unbalanced': '(()(()))('  # Note: Unbalanced for testing error handling
        }

        self.bp_trees = {}  # Store BP_Tree instances
        for name, bp_seq in self.bp_sequences.items():
            # Convert '(' to '1' and ')' to '0' for the BitVector
            bitstring = bp_seq.replace('(', '1').replace(')', '0')
            bp_tree = BP_Tree(bitstring)  # Pass bitstring, not BitVector
            self.bp_trees[name] = bp_tree  # Store BP_Tree instances

    def test_excess(self):
        # Test the 'excess' function
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        expected_excess = [1, 2, 1, 2, 3, 2, 1]
        for i, exp in enumerate(expected_excess):
            self.assertEqual(rmmtree.excess(i), exp,
                             f"Excess at position {i} should be {exp}")

    def test_depth(self):
        # Test the 'depth' function (same as 'excess')
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        expected_depth = [1, 2, 1, 2, 3, 2, 1]
        for i, exp in enumerate(expected_depth):
            self.assertEqual(rmmtree.depth(i), exp,
                             f"Depth at position {i} should be {exp}")

    def test_rmq(self):
        # Test the 'rmq' function
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree        # The minimum excess between positions 0 and 6 is at positions 0 and 6
        min_pos = rmmtree.rmq(0, 6)
        self.assertIn(min_pos, [0, 6],
                      "RMQ between 0 and 6 should be at position 0 or 6")

        # Test RMQ in a subrange
        min_pos = rmmtree.rmq(2, 5)
        self.assertEqual(min_pos, 2,
                         "RMQ between 2 and 5 should be at position 2")

    def test_rmq_max(self):
        # Test the 'rmq_max' function
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree        # The maximum excess between positions 0 and 6 is at position 4
        max_pos = rmmtree.rmq_max(0, 6)
        self.assertEqual(max_pos, 4,
                         "RMQ_MAX between 0 and 6 should be at position 4")

        # Test RMQ_MAX in a subrange
        max_pos = rmmtree.rmq_max(2, 5)
        self.assertEqual(max_pos, 4,
                         "RMQ_MAX between 2 and 5 should be at position 4")

    def test_mincount(self):
        # Test the 'mincount' function
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        count = rmmtree.mincount(0, 6)
        self.assertEqual(count, 2,
                         "Minimum excess occurs twice between positions 0 and 6")

        count = rmmtree.mincount(2, 5)
        self.assertEqual(count, 1,
                         "Minimum excess occurs once between positions 2 and 5")

    def test_minselect(self):
        # Test the 'minselect' function
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        pos = rmmtree.minselect(0, 6, 1)
        self.assertIn(pos, [0, 6],
                      "First occurrence of min excess should be at position 0 or 6")

        pos = rmmtree.minselect(0, 6, 2)
        self.assertIn(pos, [0, 6],
                      "Second occurrence of min excess should be at position 0 or 6")

        with self.assertRaises(ValueError):
            rmmtree.minselect(0, 6, 3)

    def test_fwdsearch(self):
        # Test the 'fwdsearch' function
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        idx = rmmtree.fwdsearch(0, -1)
        self.assertEqual(idx, 6,
                         "fwdsearch from position 0 with d=-1 should return 6")

        idx = rmmtree.fwdsearch(1, -2)
        self.assertEqual(idx, 6,
                         "fwdsearch from position 1 with d=-2 should return 6")

    def test_bwdsearch(self):
        # Test the 'bwdsearch' function
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        idx = rmmtree.bwdsearch(6, -1)
        self.assertEqual(idx, 0,
                         "bwdsearch from position 6 with d=-1 should return 0")

        idx = rmmtree.bwdsearch(5, -2)
        self.assertEqual(idx, 0,
                         "bwdsearch from position 5 with d=-2 should return 0")

    def test_find_close(self):
        # Test the 'find_close' function
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        idx = rmmtree.find_close(0)
        self.assertEqual(idx, 6,
                         "find_close of position 0 should return 6")

        idx = rmmtree.find_close(1)
        self.assertEqual(idx, 2,
                         "find_close of position 1 should return 2")

    def test_find_open(self):
        # Test the 'find_open' function
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        idx = rmmtree.find_open(6)
        self.assertEqual(idx, 0,
                         "find_open of position 6 should return 0")

        idx = rmmtree.find_open(2)
        self.assertEqual(idx, 1,
                         "find_open of position 2 should return 1")

    def test_enclose(self):
        # Test the 'enclose' function
        bp_tree = self.bp_trees['complex_tree']
        rmmtree = bp_tree.rmmtree
        idx = rmmtree.enclose(5)
        self.assertEqual(idx, 2,
                         "enclose of position 5 should return 2")

        idx = rmmtree.enclose(10)
        self.assertEqual(idx, 0,
                         "enclose of position 10 should return 0")

    def test_lca(self):
        # Test the 'lca' function
        bp_tree = self.bp_trees['complex_tree']
        rmmtree = bp_tree.rmmtree
        idx = rmmtree.lca(5, 9)
        self.assertEqual(idx, 2,
                         "LCA of positions 5 and 9 should be 2")

        idx = rmmtree.lca(4, 12)
        self.assertEqual(idx, 0,
                         "LCA of positions 4 and 12 should be 0")

    def test_edge_cases(self):
        # Test edge cases, including empty tree and single node
        rmmtree_empty = self.rmmtrees['empty']
        with self.assertRaises(IndexError):
            rmmtree_empty.excess(0)

        rmmtree_single = self.rmmtrees['single_node']
        self.assertEqual(rmmtree_single.excess(0), 1,
                         "Excess at position 0 should be 1 in single node tree")
        self.assertEqual(rmmtree_single.excess(1), 0,
                         "Excess at position 1 should be 0 in single node tree")

        idx = rmmtree_single.find_close(0)
        self.assertEqual(idx, 1,
                         "find_close of position 0 should return 1 in single node tree")

    def test_invalid_indices(self):
        # Test functions with invalid indices
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        with self.assertRaises(IndexError):
            rmmtree.excess(-1)

        with self.assertRaises(IndexError):
            rmmtree.excess(100)

        with self.assertRaises(IndexError):
            rmmtree.find_close(-1)

        with self.assertRaises(IndexError):
            rmmtree.find_open(100)

    def test_unbalanced_tree(self):
        # Test handling of unbalanced tree
        bp_tree = self.bp_trees['unbalanced']
        rmmtree = bp_tree.rmmtree
        with self.assertRaises(IndexError):
            rmmtree.find_close(0)

        with self.assertRaises(IndexError):
            rmmtree.find_open(7)

    def test_large_tree(self):
        # Test with a large balanced tree
        bp_seq = '(' * 1000 + ')' * 1000  # A deep left-skewed tree
        bitstring = bp_seq.replace('(', '1').replace(')', '0')
        bitvector = BitVector(bitstring)
        bp_tree = BP_Tree(bitvector)
        rmmtree = RMMinMaxTree(bp_tree)

        idx = rmmtree.find_close(0)
        self.assertEqual(idx, 1999,
                         "find_close of position 0 should return 1999 in large tree")

        depth = rmmtree.depth(500)
        self.assertEqual(depth, 501,
                         "Depth at position 500 should be 501 in large tree")

        # Test rmq_max on large tree
        max_pos = rmmtree.rmq_max(0, 1999)
        self.assertEqual(max_pos, 999,
                         "RMQ_MAX between 0 and 1999 should be at position 999")

    def test_fwdsearch_bwdsearch(self):
        # Additional tests for fwdsearch and bwdsearch
        rmmtree = self.rmmtrees['left_skewed']
        idx = rmmtree.fwdsearch(0, -4)
        self.assertEqual(idx, 8,
                         "fwdsearch from position 0 with d=-4 should return 8")

        idx = rmmtree.bwdsearch(8, 4)
        self.assertEqual(idx, 0,
                         "bwdsearch from position 8 with d=4 should return 0")

    def test_enclose_root(self):
        # Test enclose function when node is the root
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        idx = rmmtree.enclose(0)
        self.assertEqual(idx, -1,
                         "enclose of root should return -1")

    def test_isancestor(self):
        # Test the isancestor function
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        self.assertTrue(rmmtree.isancestor(0, 1),
                        "Node at position 0 should be ancestor of position 1")
        self.assertFalse(rmmtree.isancestor(1, 0),
                         "Node at position 1 should not be ancestor of position 0")

    def test_invalid_mincount(self):
        # Test mincount with invalid range
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        with self.assertRaises(IndexError):
            rmmtree.mincount(5, 100)

    def test_minselect_out_of_bounds(self):
        # Test minselect with q exceeding the number of occurrences
        bp_tree = self.bp_trees['simple_tree']
        rmmtree = bp_tree.rmmtree
        with self.assertRaises(ValueError):
            rmmtree.minselect(0, 6, 5)


if __name__ == '__main__':
    unittest.main()