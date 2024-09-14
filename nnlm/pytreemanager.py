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
        The result of the build_tree generate the head node for the dll by dict{'module':module, 'depth': depth}
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
            # Retrieve module and depth information from the current node's doubly linked list
            current_data = node.info.head.data  # .info is dll, .info.head.data is the data stored in head node
            current_module = current_data['module']  # Extract the module
            current_depth = current_data['depth']  # Extract the depth

            # If depth is defined and current depth exceeds the specified depth, stop expanding
            if depth is not None and current_depth >= depth:
                return

            # Safely check for children (submodules) in the current PyTorch module
            try:
                for child in current_module.children():
                    # Add child node to the current tree node
                    child_node = self.tree.add_child(node)

                    # Append the child module and its depth as a dictionary to the child node's dll
                    child_node.info.append({'module': child, 'depth': current_depth + 1}) # .info is dll 

            except AttributeError:
                # If the module has no children (e.g., Linear layers), pass
                pass

        # Add the root module and its depth to the root node's doubly linked list as a dictionary
        self.tree.root.info.append({'module': module, 'depth': 0})


        # Use the traverse method to build the tree
        self.tree.traverse(self.tree.root, action)

    def _merge_nodes(self, parent_node, n, m):
        """
        Merges adjacent sibling nodes from index `n` to `m` under the same parent node.
        
        Args:
            parent_node (TreeNode): The parent node containing the child nodes to be merged.
            n (int): The start index of adjacent nodes to merge.
            m (int): The end index of adjacent nodes to merge.
        """
        # Ensure indices are valid
        if not (0 <= n < len(parent_node.children) and 0 <= m < len(parent_node.children) and n <= m):
            raise ValueError("Invalid range for n and m")

        # Select the adjacent children to merge from n to m
        children_to_merge = parent_node.children[n:m + 1]

        # Check if all DoublyLinkedLists have the same length
        lengths = [len(child.info) for child in children_to_merge]
        if len(set(lengths)) != 1:
            raise ValueError("All DoublyLinkedLists must have the same length")

        # Create a new node that will be the result of the merge
        merged_node = TreeNode()

        def aggregate_action(dll_nodes_at_pos):
            """
            Aggregate function to process nodes at the same position in DLLs.
            """
            # Filter out DLL nodes that belong to children in children_to_merge
            filtered_dll_nodes = [dll_node for dll_node in dll_nodes_at_pos if any(dll_node in child.info for child in children_to_merge)]

            # If no DLL nodes belong to children_to_merge, skip this position
            if not filtered_dll_nodes:
                return

            # Access the first node's data to compare with others
            first_node_data = filtered_dll_nodes[0].data

            # Check if all the keys in the dictionaries are the same
            first_keys = first_node_data.keys()
            if not all(node.data.keys() == first_keys for node in filtered_dll_nodes):
                raise ValueError("Inconsistent keys in nodes at the same position")

            # Check if the values for all nodes at the current position are the same
            first_values = first_node_data.values()
            if all(node.data.values() == first_values for node in filtered_dll_nodes):
                # If values are the same, take the first node's values
                merged_node.info.append(first_node_data) # dll.append()
            else:
                # If 'module' key exists, merge the modules into nn.Sequential
                if 'module' in first_keys:
                    # Extract the 'module' values from the nodes at the current position
                    values = [node.data['module'] for node in filtered_dll_nodes]

                    # Ensure all 'depth' values are the same before merging
                    depths = [node.data['depth'] for node in filtered_dll_nodes]
                    if len(set(depths)) != 1:
                        raise ValueError("All 'depth' values must be the same when merging modules")

                    # Merge the modules using nn.Sequential
                    merged_module = nn.Sequential(*values)

                    # Use the first node's depth (since they should all be the same) to append to the merged node
                    merged_node.info.append({'module': merged_module, 'depth': depths[0]})
                else:
                    # For other conflicting data, set to None
                    merged_node.info.append({key: None for key in first_keys})

        # Perform sync_traverse on the selected children to merge their DLLs
        self.tree.sync_traverse(
            start_node=parent_node,
            aggregate_action=aggregate_action
        )

        # Remove the merged child nodes from the parent's list of children
        for child in children_to_merge:
            parent_node.children.remove(child)

        # Insert the new merged node into the parent's list of children at position `n`
        parent_node.children.insert(n, merged_node)
        

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
