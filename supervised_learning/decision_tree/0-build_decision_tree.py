#!/usr/bin/env python3
"""
    task 0
"""
import numpy as np


class Node:
    """
        Defines the node class
    """
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        """
            Defines the init function
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self) :
        """
            function to get the depth of the tree
        """
        ldepth = 0
        rdepth = 0
        n = self

        while n.right_child:
            rdepth = rdepth + 1
            n = n.right_child
        
        while n.left_child:
            ldepth = ldepth + 1
            n = n.left_child
        
        if rdepth > ldepth:
            return rdepth
        else:
            return ldepth

class Leaf(Node):
    """
        Defines the leaf class
    """
    def __init__(self, value, depth=None):
        """
            Defines the init function
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self) :
        """
            returns the deptth
        """
        return self.depth

class Decision_Tree():
    """
        Defines the tree class
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        """
            Defines the init function
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self) :
        """
            returns the depth of the tree
        """
        return self.root.max_depth_below()
