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

        self.root.info.append("Root")
        self.child1.info.append("Child 1")
        self.child2.info.append("Child 2")
        self.child1_1.info.append("Child 1.1")

    def test_add_child(self):
        """Test adding children to the tree"""
        self.tree.add_child(self.root)
        self.assertEqual(len(self.root.children), 3)  # Should have 3 children

    def test_delete_child(self):
        """Test deleting a child node from the tree"""
        self.tree.delete_child(self.root, self.child1)
        self.assertEqual(len(self.root.children), 1)  # Only 1 child should remain

    def test_traverse(self):
        """Test traversing the tree and collecting node info"""
        nodes = []
        
        def collect_data(node):
            nodes.append(str(node))

        self.tree.traverse(self.root, collect_data)
        self.assertEqual(nodes, ["Root", "Child 1", "Child 1.1", "Child 2"])

    def test_str(self):
        """Test the __str__ method"""
        expected = "Root\n├── Child 1\n│   ├── Child 1.1\n├── Child 2"
        self.assertEqual(str(self.tree), expected)

if __name__ == '__main__':
    unittest.main()
