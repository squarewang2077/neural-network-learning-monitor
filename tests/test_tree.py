import unittest
from nnlm.base import *

class TestTree(unittest.TestCase):

    def setUp(self):
        """Set up a tree with some nodes before each test"""
        self.tree = Tree("Test Tree")
        self.root = self.tree.root
        self.child1 = self.tree.add_child(self.root)
        self.child2 = self.tree.add_child(self.root)
        self.child1_1 = self.tree.add_child(self.child1)

        # Append data to the doubly linked list of each tree node
        self.root.info.append("Root")
        self.child1.info.append("Child 1")
        self.child2.info.append("Child 2")
        self.child1_1.info.append("Child 1.1")

        # Add more data to make sure sync_traverse handles it
        self.root.info.append("Root Additional")
        self.child1.info.append("Child 1 Additional")
        self.child2.info.append("Child 2 Additional")

    def test_sync_traverse(self):
        """Test synchronizing the traversal of each DLL across the tree nodes"""

        # Helper to aggregate and collect the data at the same positions
        aggregated_result = []

        def aggregate_action(nodes_at_position):
            # Collecting data to ensure each position is properly traversed
            aggregated_data = [str(node) for node in nodes_at_position]
            aggregated_result.append(", ".join(aggregated_data))

        # Perform the synchronized traversal starting from the root
        self.tree.sync_traverse(start_node=self.root, aggregate_action=aggregate_action)

        # Expected output for position 0 and 1
        expected = [
            "Root, Child 1, Child 1.1, Child 2",  # First element in each DLL (position 0)
            "Root Additional, Child 1 Additional, Child 2 Additional"  # Second element (position 1)
        ]

        self.assertEqual(aggregated_result, expected)


    def test_add_child(self):
        """Test adding children to the tree"""
        self.tree.add_child(self.root)
        self.assertEqual(len(self.root.children), 3)  # Should have 3 children

    def test_traverse(self):
        """Test traversing the tree and collecting node info"""
        nodes = []
        
        def collect_data(node):
            nodes.append(str(node))

        self.tree.traverse(self.root, collect_data)
        self.assertEqual(nodes, ["Root", "Child 1", "Child 1.1", "Child 2"])

    def test_str(self):
        """Test the __str__ method"""
        expected = "Root\n├── Child 1\n│   └── Child 1.1\n└── Child 2"
        self.assertEqual(str(self.tree), expected)

if __name__ == '__main__':
    unittest.main()
