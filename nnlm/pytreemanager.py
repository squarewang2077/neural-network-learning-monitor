from base import *
import torch.nn as nn  # import PyTorch module for neural network

class PyTreeManager:
    """
    Class to manage and build a tree representation of a PyTorch module
    """

    def __init__(self):
        self.tree = None

    def build_tree(self, module: nn.Module, depth=None):
        """
        Builds a tree based on the networks defined by PyTorch.
        
        Args:
            module (nn.Module): The PyTorch module to build the tree from.
            depth (int, optional): The maximum depth to build the tree. If None, build the full tree.
        """
        self.tree = Tree(tag=module.__class__.__name__)  # Initialize the tree with the module name as the tag

        def action(node):
            """
            Action function that processes each node in the tree traversal.
            Adds the PyTorch module and its depth separately into the node's doubly linked list.
            """
            current_module = node.info.head.data  # Retrieve module from the first node in the doubly linked list
            current_depth = node.info.head.next.data  # Retrieve depth from the second node in the doubly linked list

            # If depth is defined and current depth exceeds the specified depth, stop expanding
            if depth is not None and current_depth >= depth:
                return

            # Traverse through the submodules (children) of the current PyTorch module
            for child in current_module.children():
                child_node = self.tree.add_child(node)  # Add child node to the current node

                # Append the child module and its depth separately to the child node's info (DoublyLinkedList)
                child_node.info.append(child)  # Append the child module
                child_node.info.append(current_depth + 1)  # Append the depth of the child node

        # Add the root module and its depth to the tree's root node
        self.tree.root.info.append(module)  # Append the module to the root node
        self.tree.root.info.append(0)  # Append depth (0) to the root node

        # Use the traverse method to build the tree
        self.tree.traverse(self.tree.root, action)

    def _merge_nodes(self):
        pass 

    def _split_nodes(self):
        pass 


    def __str__(self):
        """String representation of the tree manager."""
        return str(self.tree) if self.tree is not None else "No tree has been built."


# Example usage with a simple PyTorch model
class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Instantiate the TreeManager and build a tree from the ExampleModel
model = ExampleModel()
tree_manager = TreeManager()
tree_manager.build_tree(model, depth=2)  # Build the tree up to depth 2
print(tree_manager)  # Print the tree structure
