#!/usr/bin/env python3
"""
    task 3
"""
import numpy as np


class Node:
    """
        Defines the node class
    """
    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
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

    def max_depth_below(self):
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

    def count_nodes_below(self, only_leaves=False):
        """
            Counts the amount of nodes
        """
        n = 1
        Leaf = 0

        if self.right_child:
            if only_leaves:
                Leaf += self.right_child.count_nodes_below(only_leaves=True)
            else:
                n = n + self.right_child.count_nodes_below()

        if self.left_child:
            if only_leaves:
                Leaf += self.left_child.count_nodes_below(only_leaves=True)
            else:
                n = n + self.left_child.count_nodes_below()

        if only_leaves and self.is_leaf:
            Leaf = Leaf + 1

        if only_leaves:
            return Leaf
        return n

    def get_leaves_below(self):
        """
            Returns a list of all leaves below this node.
        """
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            if self.left_child:
                leaves.extend(self.left_child.get_leaves_below())
            if self.right_child:
                leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def __str__(self):
        """
            def __str__(self) : method for the Node class
        """
        node_type = "root" if self.is_root else "-> node"

        node_repr = f"{node_type} [feature={self.feature},\
 threshold={self.threshold}]\n"

        if self.left_child:
            node_repr +=\
                self.left_child_add_prefix(self.left_child.__str__().strip())

        if self.right_child:
            node_repr +=\
                self.right_child_add_prefix(self.right_child.__str__().strip())

        return node_repr

    def left_child_add_prefix(self, text):
        """
            add lchild prefix
        """
        lines = text.split("\n")
        new_text = "    +--"+lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  "+x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
            add rchild prefix
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text


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

    def max_depth_below(self):
        """
            returns the deptth
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
            returns the amount of nodes of a leaf (1)
        """
        return 1

    def get_leaves_below(self):
        """
            Returns a list containing itself since it's a leaf
        """
        return [self]

    def __str__(self):
        """
            __str__(self) : method for the Decision_Tree class:
        """
        return (f"-> leaf [value={self.value}] ")


class Decision_Tree():
    """
        Defines the tree class
    """
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
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

    def depth(self):
        """
            returns the depth of the tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
            returns the amount of nodes of the tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """
            Returns the list of all leaves in the tree
        """
        return self.root.get_leaves_below()

    def __str__(self):
        """
            __str__(self) : method for the Decision_Tree class:
        """
        return self.root.__str__()
