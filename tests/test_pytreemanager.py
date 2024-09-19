import unittest
import torch
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
        self.layer3 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # Reshape x to match the input shape expected by Conv2d
        x = x.view(x.size(0), 1, 3, 1)  # Assuming batch size is the first dimension
        x = self.layer3(x)
        return x

class ReshapeLayer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 1, 3, 1)

class ComplexModel2(nn.Module):
    def __init__(self):
        super(ComplexModel2, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(10, 5))
        self.layer2 = nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU()
        )
        self.reshape = nn.Sequential(ReshapeLayer())  # Use the custom ReshapeLayer
        self.layer3 = nn.Sequential(nn.Conv2d(1, 1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.reshape(x)  # Reshape x to match the input shape expected by Conv2d
        x = self.layer3(x)
        return x


class ComplexModel3(nn.Module):
    def __init__(self):
        super(ComplexModel3, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(10, 5))
        self.layer2 = nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(nn.Conv2d(1, 1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # Reshape x to match the input shape expected by Conv2d
        x = x.view(x.size(0), 1, 3, 1)  # Assuming batch size is the first dimension
        x = self.layer3(x)
        return x

class TestPyTreeManagerComplex(unittest.TestCase):
    def setUp(self):
        """
        Set up a complex model and PyTreeManager for each test case.
        """
        self.model = ComplexModel()  # A more complex PyTorch model with multiple layers and sequential blocks
        self.model2 = ComplexModel2()  # The model with the all the leaves having the same depth
        self.model3 = ComplexModel3()  # The model to test the error handling of feature_tracker
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

    def test_add_alias(self):
        """
        Test if the _add_alias method correctly assigns unique names to the head node of each DLL in the tree.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Apply the add_alias function
        self.manager._alias("add")

        # Check the alias of each node in the tree
        def check_alias(node):
            dll_node = node.info.head
            if hasattr(dll_node, 'module') and hasattr(dll_node, 'depth'):
                module_name = dll_node.module.__class__.__name__
                depth = dll_node.depth
                expected_alias = f"{depth}:{module_name}"
                self.assertEqual(dll_node.alias, expected_alias, f"Alias should be '{expected_alias}' but got '{dll_node.alias}'.")

        # Traverse the tree and check aliases
        self.manager.tree.traverse(self.manager.tree.root, check_alias)

    def test_remove_alias(self):
        """
        Test if the _alias method correctly removes the unique names from the head node of each DLL in the tree.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Apply the add_alias function to add aliases first
        self.manager._alias(add_or_remove="add")

        # Now, apply the remove_alias function to remove the aliases
        self.manager._alias(add_or_remove="remove")

        # Check that the alias of each node in the tree is removed
        def check_alias_removed(node):
            dll_node = node.info.head
            if hasattr(dll_node, 'module') and hasattr(dll_node, 'depth'):
                self.assertFalse(hasattr(dll_node, 'alias'), "Alias should be removed from the DLL node.")

        # Traverse the tree and check aliases
        self.manager.tree.traverse(self.manager.tree.root, check_alias_removed)

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
        
        # Check the children, the leaf node should not be removed  
        self.assertEqual(len(self.manager.tree.root.children[0].children), 0, "Nodes at depth 1 should have 2 leaf children after pruning.")
        self.assertEqual(len(self.manager.tree.root.children[1].children), 2, "Nodes at depth 1 should have 2 leaf children after pruning.")
        self.assertEqual(len(self.manager.tree.root.children[2].children), 0, "Nodes at depth 1 should have 2 leaf children after pruning.")

    def test_prune_tree_with_leaf_nodes(self):
        """
        Test if the prune function correctly retains leaf nodes even if they are not in the retained depths.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Before pruning, the tree should have the root and its 3 children
        retained_depths = [0]  # We want to keep only the root

        # Manually add a leaf node to the first child to test leaf node retention
        leaf_node = TreeNode()
        leaf_dll_node = DoublyListNode()
        leaf_dll_node.module = nn.Linear(5, 2)
        leaf_dll_node.depth = 2
        leaf_node.info.append(leaf_dll_node)
        self.manager.tree.root.children[0].children.append(leaf_node)

        self.manager.prune(retained_depths)

        # Verify that the root has only one child (the first child with the leaf node)
        self.assertEqual(len(self.manager.tree.root.children), 4, "Root should have 4 children after pruning.")

    def test_get_max_depth(self):
        """
        Test if the _get_max_depth method correctly returns the maximum depth of the tree.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Get the maximum depth of the tree
        max_depth = self.manager._get_max_depth()

        # Verify that the maximum depth is correct
        self.assertEqual(max_depth, 2, "The maximum depth of the tree should be 2.")

    def test_feature_tracker(self):
        """
        Test the feature_tracker method to ensure it correctly tracks and stores features.
        """
        # Define a simple input tensor
        batched_inputs = torch.randn(2, 10)  # Batch size of 2, input size of 10

        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model2)

        # Apply the feature tracker
        self.manager.feature_tracker(batched_inputs)

        # Verify that the features are stored correctly in the DLL nodes
        def check_features(node):
            dll_node = node.info.head
            if hasattr(dll_node, 'batched_input_features') and hasattr(dll_node, 'batched_output_features'):
                # Check that the input and output features are stored
                self.assertIsNotNone(dll_node.batched_input_features, "Input features should be stored in the DLL node.")
                self.assertIsNotNone(dll_node.batched_output_features, "Output features should be stored in the DLL node.")

                # Check that the input features match the expected shape
                for input_features in dll_node.batched_input_features:
                    self.assertEqual(input_features.shape, batched_inputs.shape, "Input features should have the same shape as the batched inputs.")

                # Check that the output features are computed correctly
                for output_features in dll_node.batched_output_features:
                    self.assertEqual(output_features.shape[0], batched_inputs.shape[0], "Output features should have the same batch size as the batched inputs.")

        # Traverse the tree and check features
        self.manager.tree.traverse(self.manager.tree.root, check_features)

    def test_feature_tracker_module_error(self):
        """
        Test if the feature_tracker method raises an error when a module fails to compute output features.
        """
      
        # Build the tree from the faulty model
        self.manager.build_tree(self.model3)

        # Define a simple input tensor
        batched_inputs = torch.randn(2, 10)  # Batch size of 2, input size of 10

        # Apply the feature tracker and expect a RuntimeError
        with self.assertRaises(RuntimeError) as context:
            self.manager.feature_tracker(batched_inputs)

        self.assertTrue("Failed to compute output features for module" in str(context.exception))

    def test_get_attr(self):
        """
        Test if the _get_attr method correctly retrieves the attribute value of the DLL nodes in the tree.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model)

        # Define a function to add a custom attribute to each DLL node
        def add_custom_attr(tree_node):
            dll_node = DoublyListNode()
            dll_node.custom_attr = "test_value"
            tree_node.info.append(dll_node)

        # Traverse the tree and add the custom attribute to each node
        self.manager.tree.traverse(self.manager.tree.root, add_custom_attr)

        # Retrieve the custom attribute value using _get_attr
        attr_node, attr_values = self.manager._get_attr(self.manager.tree.root, 'custom_attr')

        # Verify that the retrieved attribute values are correct
        self.assertEqual(attr_values, "test_value", "The custom attribute value should be 'test_value'.")
        self.assertEqual(attr_node, self.manager.tree.root.info.tail, "The custom attribute node should be from the root node.")

    def test_expand_leaves(self):
        """
        Test if the _expand_leaves method correctly expands all the leaves having .children attribute in the head of the DLL node.
        """
        # Build the tree from the PyTorch model
        self.manager.build_tree(self.model, depth = 1)

        # Before expanding leaves, the root should have 3 children
        self.assertEqual(len(self.manager.tree.root.children), 3, "Root should have 3 children before expanding leaves.")

        # Apply the _expand_leaves method
        self.manager._expand_leaves()

        # After expanding leaves, the second child (Sequential block) should have 2 children
        self.assertEqual(len(self.manager.tree.root.children[1].children), 2, "Second child should have 2 children after expanding leaves.")

        # Verify that the children of the second child are the Linear and ReLU layers
        self.assertIsInstance(self.manager.tree.root.children[1].children[0].info.head.module, nn.Linear, "First child of the second child should be a Linear layer.")
        self.assertIsInstance(self.manager.tree.root.children[1].children[1].info.head.module, nn.ReLU, "Second child of the second child should be a ReLU layer.")

        # Verify that the depth of the expanded children is correct
        self.assertEqual(self.manager.tree.root.children[1].children[0].info.head.depth, 2, "Depth of the first child of the second child should be 2.")
        self.assertEqual(self.manager.tree.root.children[1].children[1].info.head.depth, 2, "Depth of the second child of the second child should be 2.")
if __name__ == '__main__':
    unittest.main()
