# bp

Implementation of Navarro and Cordova's "Simple and Efficient Fully-Functional Succinct Trees" (2018). Builds off of biocore's [improved-octo-waddle](https://github.com/biocore/improved-octo-waddle/tree/master). Work in progress. 

Next steps:
- Check that my implementation aligns with paper
- More robust and comprehensive unit testing
- More extensive benchmarking
- Convert to Cython
- Leaf operations

# SuccinctTree

A **SuccinctTree** implementation in Python that efficiently supports all common tree navigation and query operations in **O(log log n)** time and uses **2n + O(n/log n)** bits of space via a balanced parentheses (BP) encoding.

## Features

- **Rank/Select Operations**: `rmq`, `rMq`, `mincount`, `minselect`, `count_excess` in O(log log n).
- **Matching & Navigation**: `close`, `open`, `enclose`, `parent`, `children`, `siblings` in O(log log n).
- **Traversal Orders**: `preorder`, `postorder`, `preorderselect`, `postorderselect` in O(1) or O(log log n).
- **Tree Queries**: `depth`, `height`, `degree`, `subtree`, `lca`, `deepestnode`, `ntips` in O(log log n).
- **Level Operations**: `levelancestor`, `levelnext`, `levelprev`, `levelleftmost`, `levelrightmost` in O(log log n).

- `'('` (or `1`) marks an **opening** parenthesis for a node.
- `')'` (or `0`) marks a **closing** parenthesis for a node.

> **Dependencies**: `numpy`, `sdsl4py` (for succinct bitvector support).

## Usage

```python
from succinct import SuccinctTree

# Build from balanced-parentheses string:
bp_str = '((())(()()))'
tree = SuccinctTree(bp_str)

# Or from a 0/1 array:
# bits = np.array([1,1,1,0,0,1,1,0,0,1,0,0])
# tree = SuccinctTree(bits)

# Queries:
root = tree.root()
child0 = tree.fchild(root)
num_nodes = tree.subtree(root)
print('Subtree size:', num_nodes)

# Orderings:
print('Preorder of root:', tree.preorder(root))
print('Deepest node:', tree.deepestnode(root))
```

### Basic Operations

- `close(i)`, `open(i)`, `enclose(i)`
- `rmq(i,j)`, `rMq(i,j)`, `mincount(i,j)`, `minselect(i,j,q)`
- `fwdsearch(i,d)`, `bwdsearch(i,d)`
- `rank1(pos)`, `select1(k)`, `rank0(pos)`, `select0(k)`

### Tree Navigation

- `parent(i)`, `fchild(i)`, `lchild(i)`, `nsibling(i)`, `psibling(i)`
- `depth(i)`, `height(i)`, `degree(i)`, `subtree(i)`, `ntips()`
- `lca(i,j)`, `deepestnode(i)`
- `levelancestor(i,d)`, `levelnext(i)`, `levelprev(i)`, `levelleftmost(d)`, `levelrightmost(d)`

### Metadata

- `set_names(names: Iterable)`, `name(i)`
- `set_lengths(lengths: Iterable)`, `length(i)`

### Transformations

- `collapse() -> SuccinctTree`
- `shear(tip_set: Set[str]) -> SuccinctTree`

## Performance

- **Time complexity**: All navigation and query operations run in **O(log log n)** worst-case (or O(1) for pure rank/select).
- **Space complexity**: Uses **2n + o(n)** bits for the BP bitvector plus **O(n/log n)** auxiliary structures.

## Testing

Run the full unit test suite from project root with:

```bash
pytest -q
```

or 

```bash
python -m unittest discover tests
```

Tests from `test_bp.py` check correctness using example from paper.

---

*This implementation follows J. Cordova and G. Navarro’s “Simple and Efficient Fully-Functional Succinct Trees” (2018).*
