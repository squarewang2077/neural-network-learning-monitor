import unittest
from nnlm.base import *

class TestTreeNode(unittest.TestCase):
    def test_tree_node_initialization(self):
        """
        Test that a TreeNode is initialized properly with an empty DoublyLinkedList and no children.
        """
        tree_node = TreeNode()
        self.assertIsInstance(tree_node.info, DoublyLinkedList) # test whether treenode.info is DLL
        self.assertEqual(tree_node.children, []) # test whether treenode.children is a list

    def test_tree_node_string_representation(self):
        """
        Test the string representation of the TreeNode.
        When the node is created, it should be "Empty" since its DoublyLinkedList is empty.
        """
        tree_node = TreeNode()
        self.assertEqual(str(tree_node), "Empty") # tree_node.info.head = None, __str__ will give the head info

        # initilize the DLL node
        dll_node = DoublyListNode()
        dll_node.data = 'Head' 

        tree_node.info.append(dll_node) # append the dll node on the tree_node.info
        self.assertEqual(str(tree_node), "data: Head") # the tree_node.head.info 

class TestTree(unittest.TestCase):
    def setUp(self):
        """
        Set up a basic Tree for testing.
        """
        self.tree = Tree(tag="test_tree")

    def test_tree_initialization(self):
        """
        Test that the Tree is initialized properly with a root TreeNode and a tag.
        """
        self.assertIsInstance(self.tree.root, TreeNode)
        self.assertEqual(self.tree.tag, "test_tree")

    def test_add_child(self):
        """
        Test that a child node is correctly added to the parent node.
        """
        parent = self.tree.root
        child = self.tree.add_child(parent)
        self.assertIn(child, parent.children)
        self.assertEqual(len(parent.children), 1)

    def test_traverse(self):
        """
        Test the pre-order traversal of the tree. The action function will collect the nodes
        in the order they are traversed.
        """
        parent = self.tree.root
        child1 = self.tree.add_child(parent)
        child2 = self.tree.add_child(parent)

        child1_1 = self.tree.add_child(child1)

        visited_nodes = []

        def collect_action(node):
            visited_nodes.append(node)

        self.tree.traverse(action=collect_action)

        self.assertEqual(visited_nodes, [parent, child1, child1_1, child2])

    def test_sync_traverse(self):
        """
        Test the sync_traverse method to ensure it correctly synchronizes traversal across all nodes' DoublyLinkedLists.
        We will simply verify that the aggregate_action is called the correct number of times.
        """
        # Set up the DLLs in each node
        node1 = DoublyListNode()
        node1.data = 'node1'
        node2 = DoublyListNode()
        node2.data = 'node2'
        node3 = DoublyListNode()
        node3.data = 'node3'

        root = self.tree.root
        child1 = self.tree.add_child(root)
        child2 = self.tree.add_child(root)

        root.info.append(node1)
        child1.info.append(node2)
        child2.info.append(node3)

        sync_results = []

        def aggregate_action(nodes_at_position):
            sync_results.append([str(node) for node in nodes_at_position])

        self.tree.sync_traverse(aggregate_action=aggregate_action)

        # Check that the aggregate_action was called once for each position and collected data correctly
        self.assertEqual(sync_results, [['data: node1', 'data: node2', 'data: node3']])

    def test_tree_str(self):
        """
        Test the string representation of the tree.
        It should print the tree structure with proper indentation and lines.
        """
        parent = self.tree.root
        child1 = self.tree.add_child(parent)
        child2 = self.tree.add_child(parent)
        child1_1 = self.tree.add_child(child1)

        # Set up the DLLs in each node
        node0 = DoublyListNode()
        node0.data = 'parent'

        node1 = DoublyListNode()
        node1.data = 'child1'
        
        node2 = DoublyListNode()
        node2.data = 'child2'
        
        node3 = DoublyListNode()
        node3.data = 'child1_1'

        parent.info.append(node0)
        child1.info.append(node1)
        child2.info.append(node2)
        child1_1.info.append(node3)

        expected_output = (
            "data: parent\n"
            "├── data: child1\n"
            "│   └── data: child1_1\n"
            "└── data: child2"
        )

        self.assertEqual(str(self.tree), expected_output)

if __name__ == '__main__':
    unittest.main()
