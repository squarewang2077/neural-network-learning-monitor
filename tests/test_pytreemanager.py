import unittest
import torch.nn as nn
from nnlm.pytreemanager import *

# Define a more complex PyTorch model for testing
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU()
        )
        self.layer3 = nn.Conv2d(1, 1, 3)

class TestPyTreeManagerComplex(unittest.TestCase):
    def setUp(self):
        """
        Set up a complex model and PyTreeManager for each test case.
        """
        self.model = ComplexModel()  # A more complex PyTorch model with multiple layers and sequential blocks
        self.manager = PyTreeManager()  # Instantiate the tree manager

    def test_build_tree_complex(self):
        """
        Test if the tree is built correctly based on the complex PyTorch model.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Check if the tree is built with the correct tag
        self.assertEqual(self.manager.tree.tag, 'ComplexModel', "Tree root tag should be 'ComplexModel'.")

        # Verify that the root node has children (the layers of the model)
        self.assertEqual(len(self.manager.tree.root.children), 3, "Root node should have 3 children (3 layers).")

        # Check that each child node has the correct module stored in the doubly linked list
        for idx, child in enumerate(self.manager.tree.root.children):
            dll_node = child.info.head
            if idx == 0:
                self.assertIsInstance(dll_node.module, nn.Linear, "First child should be a Linear layer.")
            elif idx == 1:
                self.assertIsInstance(dll_node.module, nn.Sequential, "Second child should be a Sequential block.")
            elif idx == 2:
                self.assertIsInstance(dll_node.module, nn.Conv2d, "Third child should be a Conv2d layer.")

    def test_prune_tree_complex(self):
        """
        Test if the prune function correctly removes nodes not in the retained depths for a complex model.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Before pruning, the tree should have the root and its 3 children
        retained_depths = [0, 1]  # We want to keep the root and its immediate children (depth 0 and 1)
        self.manager.prune(retained_depths)

        # Verify that only root and its children (depth 1) are kept
        self.assertEqual(len(self.manager.tree.root.children), 3, "There should still be 3 children after pruning.")
        
        # Check if any child of the root has children (should not have since only depth 1 is retained)
        for child in self.manager.tree.root.children:
            self.assertEqual(len(child.children), 0, "Nodes at depth 1 should have no children after pruning.")

    def test_find_parent_node(self):
        """
        Test the _find_parent_node method to find the parent of a given node.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Get the second child (layer2)
        target_node = self.manager.tree.root.children[1]

        # Find the parent node
        parent_node = self.manager._find_parent_node(target_node)

        # Check that the parent node is the root
        self.assertEqual(parent_node, self.manager.tree.root, "The parent of the second layer should be the root node.")

    def test_remove_node(self):
        """
        Test the _remove_node method by removing a node from the tree.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Remove the second child (layer2)
        node_to_remove = self.manager.tree.root.children[1]
        self.manager._remove_node(node_to_remove)

        # Verify that the node was removed
        self.assertEqual(len(self.manager.tree.root.children), 4, "After removing the mid-node, root should have 4 children.")

        # Check that the removed node's position was filled by its children 
        self.assertEqual(self.manager.tree.root.children[0].info.head.module, self.model.layer1,
                         "First child should still be layer1 after removal.")
        self.assertEqual(self.manager.tree.root.children[1].info.head.module, [m for m in self.model.layer2.children()][0],
                         "Second child should now be layer2 sub modual of Linear(5, 3).")
        self.assertEqual(self.manager.tree.root.children[2].info.head.module, [m for m in self.model.layer2.children()][1],
                         "Second child should now be layer2 sub modual of ReLU.")
        self.assertEqual(self.manager.tree.root.children[3].info.head.module, self.model.layer3,
                         "Second child should now be layer3 after removal of layer3.")

    def test_merge_nodes(self):
        """
        Test if the merge function correctly merges adjacent nodes in the complex model.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Merge the first and second children of the root
        self.manager._merge_nodes(self.manager.tree.root, 0, 1)

        # After merging, there should be 2 children under the root
        self.assertEqual(len(self.manager.tree.root.children), 2, "After merging, root should have 2 children.")
        
        # Check that the merged node has an nn.Sequential module with the two layers (Linear and Sequential)
        merged_node = self.manager.tree.root.children[0].info.head
        self.assertIsInstance(merged_node.module, nn.Sequential, "The merged node should contain an nn.Sequential module.")
        self.assertEqual(len(merged_node.module), 2, "The nn.Sequential module should have 2 layers after merging.")

    def test_replicate_info(self):
        """
        Test if the _replicate_info method correctly replicates the information of a given node.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Replicate the info of the first child node
        node_to_replicate = self.manager.tree.root.children[0]
        replicated_node = self.manager._replicate_info(node_to_replicate)

        # Verify that the replicated node has the same structure, but all attributes (except next and prev) are None
        original_dll = node_to_replicate.info.head
        replicated_dll = replicated_node.info.head

        # Check that the 'module' and 'depth' in the replicated node are None
        self.assertIsNone(replicated_dll.module, "The module in the replicated node should be None.")
        self.assertIsNone(replicated_dll.depth, "The depth in the replicated node should be None.")

        # Check whether the length of the replicated_dll is tha same with original_dll
        self.assertEqual(len(replicated_node.info), len(node_to_replicate.info))

    def test_insert_node_after(self):
        """
        Test if the _insert_node_after method correctly inserts a replicated node after a given node.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Insert a new node after the first child
        node_to_insert_after = self.manager.tree.root.children[0]
        self.manager._insert_node_after(node_to_insert_after)

        # Verify that a new node was inserted after the first child
        self.assertEqual(len(self.manager.tree.root.children), 4, "After insertion, root should have 4 children.")
        
        # Verify that the new node has no module or depth
        inserted_node = self.manager.tree.root.children[1]
        self.assertIsNone(inserted_node.info.head.module, "The inserted node should have no module.")
        self.assertIsNone(inserted_node.info.head.depth, "The inserted node should have no depth.")

if __name__ == '__main__':
    unittest.main()
