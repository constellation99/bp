import unittest
import numpy as np
import numpy.testing as npt
import os, sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
)

from rmm_tree import BucketRMMinMax
from succinct_tree import SuccinctTree


class SuccinctTreeTests(unittest.TestCase):
    def setUp(self):
        # fig1 example bitvector
        self.fig1_B = np.array([
            1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1,
            0, 1, 1, 1, 0, 1, 0, 0, 0, 0
        ], dtype=np.uint8)
        # convert to parentheses string
        bp_str = ''.join('(' if b else ')' for b in self.fig1_B)
        self.tree = SuccinctTree(bp_str)

    def test_rmq(self):
        exp = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 21],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                     [2, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                        [3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                           [4, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                              [5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                 [6, 6, 6, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                    [7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                       [8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                          [9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                             [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21],
                                                 [11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 21],
                                                     [12, 12, 12, 12, 12, 12, 12, 12, 12, 21],
                                                         [13, 13, 13, 13, 13, 13, 13, 20, 21],
                                                             [14, 14, 14, 14, 14, 19, 20, 21],
                                                                 [15, 16, 16, 16, 19, 20, 21],
                                                                     [16, 16, 16, 19, 20, 21],
                                                                         [17, 18, 19, 20, 21],
                                                                             [18, 19, 20, 21],
                                                                                 [19, 20, 21],
                                                                                     [20, 21],
                                                                                         [21]]
        n = len(self.fig1_B)
        for i in range(n):
            for j in range(i+1, n):
                self.assertEqual(self.tree.rmq(i, j), exp[i][j - i])

    def test_rMq(self):
        #       (  (  (  )  (  )  (  (  )  )   )   (   )   (   (   (   )   (   )   )   )   )
        #excess 1  2  3  2  3  2  3  4  3  2   1   2   1   2   3   4   3   4   3   2   1   0
        #i      0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21

        exp = [[0, 1, 2, 2, 2, 2, 2, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                  [1, 2, 2, 2, 2, 2, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                     [2, 2, 2, 2, 2, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                        [3, 4, 4, 4, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                           [4, 4, 4, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                              [5, 6, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                                 [6, 7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                                    [7, 7, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 7],
                                       [8, 8,  8,  8,  8,  8,  8, 15, 15, 15, 15, 15, 15, 15],
                                          [9,  9,  9,  9,  9, 14, 15, 15, 15, 15, 15, 15, 15],
                                             [10, 11, 11, 11, 14, 15, 15, 15, 15, 15, 15, 15],
                                                 [11, 11, 11, 14, 15, 15, 15, 15, 15, 15, 15],
                                                     [12, 13, 14, 15, 15, 15, 15, 15, 15, 15],
                                                         [13, 14, 15, 15, 15, 15, 15, 15, 15],
                                                             [14, 15, 15, 15, 15, 15, 15, 15],
                                                                 [15, 15, 15, 15, 15, 15, 15],
                                                                     [16, 17, 17, 17, 17, 17],
                                                                         [17, 17, 17, 17, 17],
                                                                             [18, 18, 18, 18],
                                                                                 [19, 19, 19],
                                                                                     [20, 20],
                                                                                         [21]]
        n = len(self.fig1_B)
        for i in range(n):
            for j in range(i+1, n):
                self.assertEqual(self.tree.rMq(i, j), exp[i][j - i])

    def test_mincount(self):
        exp = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4, 1],
                  [1, 1, 2, 2, 3, 3, 3, 3, 4,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                     [1, 1, 1, 2, 2, 2, 2, 3,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                        [1, 1, 2, 2, 2, 2, 3,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                           [1, 1, 1, 1, 1, 2,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                              [1, 1, 1, 1, 2,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                 [1, 1, 2, 1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                    [1, 1, 1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                       [1, 1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                          [1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                              [1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3, 1],
                                                  [1,  1,  1,  1,  1,  1,  1,  1,  1,  2, 1],
                                                      [1,  1,  1,  1,  1,  1,  1,  1,  2, 1],
                                                          [1,  1,  1,  1,  1,  1,  2,  1, 1],
                                                              [1,  1,  2,  2,  3,  1,  1, 1],
                                                                  [1,  1,  1,  2,  1,  1, 1],
                                                                      [1,  1,  2,  1,  1, 1],
                                                                          [1,  1,  1,  1, 1],
                                                                              [1,  1,  1, 1],
                                                                                  [1,  1, 1],
                                                                                      [1, 1],
                                                                                         [1]]
        n = len(self.fig1_B)
        for i in range(n):
            for j in range(i+1, n):
                self.assertEqual(self.tree.mincount(i, j), exp[i][j - i])

    def test_minselect(self):
        exp = {(0, 20, 1): 0,
               (0, 21, 1): 21,
               (0, 20, 2): 10,
               (0, 21, 2): None,
               (0, 20, 3): 12,
               (0, 20, 4): 20,
               (8, 15, 1): 10,
               (8, 15, 2): 12,
               (6, 9, 1): 9}
        for (i,j,q), e in exp.items():
            self.assertEqual(self.tree.minselect(i, j, q), e)

    def test_preorder(self):
        exp = [1,2,3,3,4,4,5,6,6,5,2,7,7,8,9,10,10,11,11,9,8,1]
        for i, e in enumerate(exp):
            self.assertEqual(self.tree.preorder(i), e)

    def test_preorderselect(self):
        exp = [0,1,2,4,6,7,11,13,14,15,17]
        for k, e in enumerate(exp):
            self.assertEqual(self.tree.preorderselect(k), e)

    def test_postorder(self):
        exp = [11,5,1,1,2,2,4,3,3,4,5,6,6,10,9,7,7,8,8,9,10,11]
        for i, e in enumerate(exp):
            self.assertEqual(self.tree.postorder(i), e)

    def test_postorderselect(self):
        exp = [2,4,7,6,1,11,15,17,14,13,0]
        for k, e in enumerate(exp):
            self.assertEqual(self.tree.postorderselect(k+1), e)

    def test_isancestor(self):
        exp = {
            (0,0): False,
            (2,1): False,
            (1,2): True,
            (1,3): True,
            (0,7): True,
            (1,7): True
        }
        for (i,j), e in exp.items():
            self.assertEqual(self.tree.isancestor(i, j), e)

    def test_subtree(self):
        exp = [11,5,1,1,1,1,2,1,1,2,5,1,1,4,3,1,1,1,1,3,4,11]
        for i, e in enumerate(exp):
            self.assertEqual(self.tree.subtree(i), e)

    def test_levelancestor(self):
        exp = {
            (2,1): 1,
            (2,2): 0,
            (4,1): 1,
            (5,1): 1,
            (7,1): 6,
            (7,2): 1,
            (7,3): 0,
            (7,9999): 0,
            (10,0): -1
        }
        for (i,d), e in exp.items():
            self.assertEqual(self.tree.levelancestor(i, d), e)

    def test_levelnext(self):
        exp = [-1,11,4,4,6,6,14,15,15,14,11,13,13,-1,-1,17,17,-1,-1,-1,-1,-1]
        n = len(self.fig1_B)
        self.assertEqual(len(exp), n)
        for i, e in enumerate(exp):
            self.assertEqual(self.tree.levelnext(i), e)

    def test_close(self):
        exp = [21,10,3,5,9,8,12,20,19,16,18]
        # positions of '(' in fig1_B
        opens = np.argwhere(self.tree.B == 1).squeeze()
        for i, e in zip(opens, exp):
            npt.assert_equal(self.tree.close(i), e)

    def test_lca(self):
        # pick nodes via preorderselect
        nodes = [self.tree.preorderselect(k) for k in range(self.fig1_B.sum())]
        exp = {
            (nodes[2], nodes[3]): nodes[1],
            (nodes[2], nodes[5]): nodes[1],
            (nodes[2], nodes[9]): nodes[0],
            (nodes[9], nodes[10]): nodes[8],
            (nodes[1], nodes[8]): nodes[0]
        }
        for (i, j), e in exp.items():
            self.assertEqual(self.tree.lca(i, j), e)

    def test_deepestnode(self):
        exp = [7,7,2,2,4,4,7,7,7,7,7,11,11,15,15,15,15,17,17,15,15,7]
        for i, e in enumerate(exp):
            self.assertEqual(self.tree.deepestnode(i), e)

    def test_height(self):
        exp = [3,2,0,0,0,0,1,0,0,1,2,0,0,2,1,0,0,0,0,1,2,3]
        for i, e in enumerate(exp):
            self.assertEqual(self.tree.height(i), e)

    def test_ntips(self):
        self.assertEqual(self.tree.ntips(), 6)

    def test_shear(self):
        names = np.array(['r','2','3',None,'4',None,'5','6',None,None,None,'7',None,'8','9','10',None,'11',None,None,None,None])
        lengths = np.array([0,1,2,0,3,0,4,5,0,0,0,6,0,7,8,9,0,10,0,0,0,0], dtype=float)
        self.tree.set_names(names)
        self.tree.set_lengths(lengths)
        in_set = {'4','6','7','10','11'}
        obs = self.tree.shear(in_set)
        expB = np.array([1,1,1,0,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,0], dtype=np.uint32)
        npt.assert_equal(obs.B, expB)
        exp_n = ['r','2','4',None,'5','6',None,None,None,'7',None,'8','9','10',None,'11',None,None,None,None]
        exp_l = [0,1,3,0,4,5,0,0,0,6,0,7,8,9,0,10,0,0,0,0]
        for i in range(len(obs.B)):
            self.assertEqual(obs.name(i), exp_n[i])
            self.assertEqual(obs.length(i), exp_l[i])
        # second shear
        obs2 = obs.shear({'10','11'})
        exp2 = np.array([1,1,1,1,0,1,0,0,0,0], dtype=np.uint32)
        npt.assert_equal(obs2.B, exp2)

    def test_shear_raise_tree_is_empty(self):
        names = np.array(['r','2','3',None,'4',None,'5','6',None,None,None,'7',None,'8','9','10',None,'11',None,None,None,None])
        self.tree.set_names(names)
        with self.assertRaises(ValueError):
            self.tree.shear({'not','in','tree'})

    def test_collapse(self):
        names = np.array(['r','2','3',None,'4',None,'5','6',None,None,None,'7',None,'8','9','10',None,'11',None,None,None,None])
        lengths = np.array([0,1,2,0,3,0,4,5,0,0,0,6,0,7,8,9,0,10,0,0,0,0], dtype=float)
        self.tree.set_names(names)
        self.tree.set_lengths(lengths)
        obs = self.tree.collapse()
        expB = np.array([1,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,0,0], dtype=np.uint8)
        npt.assert_equal(obs.B, expB)
        exp_n = ['r','2','3',None,'4',None,'6',None,None,'7',None,'9','10',None,'11',None,None,None]
        exp_l = [0,1,2,0,3,0,9,0,0,6,0,15,9,0,10,0,0,0]
        for i in range(len(obs.B)):
            self.assertEqual(obs.name(i), exp_n[i])
            self.assertEqual(obs.length(i), exp_l[i])
        # collapse small tree
        small_bp = SuccinctTree('((())())')
        small_obs = small_bp.collapse()
        npt.assert_equal(small_obs.B, np.array([1,1,0,1,0,0]))

    def test_name_unset(self):
        for i in range(len(self.tree.B)):
            self.assertIsNone(self.tree.name(i))

    def test_length_unset(self):
        for i in range(len(self.tree.B)):
            self.assertEqual(self.tree.length(i), 0.0)

    def test_name_length_set(self):
        # default arrays
        names = np.full(len(self.tree.B), None, dtype=object)
        lengths = np.zeros(len(self.tree.B), dtype=float)
        names[0] = 'root'
        names[self.tree.preorderselect(7)] = 'other'
        lengths[1] = 1.23
        lengths[self.tree.preorderselect(5)] = 5.43
        self.tree.set_names(names)
        self.tree.set_lengths(lengths)
        self.assertEqual(self.tree.name(0), 'root')
        self.assertIsNone(self.tree.name(1))
        self.assertEqual(self.tree.name(self.tree.preorderselect(7)), 'other')
        self.assertEqual(self.tree.length(1), 1.23)
        self.assertEqual(self.tree.length(self.tree.preorderselect(5)), 5.43)

if __name__ == '__main__':
    unittest.main()
