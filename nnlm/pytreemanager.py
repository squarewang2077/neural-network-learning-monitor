from base import *
import torch.nn as nn  

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
            # If depth is defined and current depth exceeds the specified depth, stop expanding
            if depth is not None and node.info.depth >= depth: 
                return

            # Safely check for children (submodules) in the current PyTorch module
            try:
                for child in node.info.module.children():
            
                    child_node = self.tree.add_child(node) # Add child node to the current tree node

                    # Add info for the child node 
                    child_dll_node = DoublyListNode() # the dll node to store the infomation 
                    child_dll_node.module = child 
                    child_dll_node.depth = node.info.depth + 1

                    child_node.info.append(child_dll_node) # append on the child_node.info
            
            # If the module has no children (e.g., Linear layers), pass
            except AttributeError:
                pass

        # Add the root module and its depth to the root node's doubly linked list as a dictionary
        root_dll_node = DoublyListNode()
        root_dll_node.module = module
        root_dll_node.depth = 0 
        self.tree.root.info.append(root_dll_node)


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
            # The dict of all attributes, note that it contains at least {'next': ..., 'prev': ....}
            first_dll_node_attributes = filtered_dll_nodes[0].__dict__ 

            # Check if all attributes' name are the same at the same position of all dll nodes for different tree nodes  
            first_keys = first_dll_node_attributes.keys()
            if not all(dll_node.__dict__.keys() == first_keys for dll_node in filtered_dll_nodes):
                raise ValueError("Inconsistent keys in nodes at the same position")

            # The function that filter out 'next' and 'prev' attributes from the dictionaries' values before comparison
            def filter_basic_attribute(d):
                return {k: v for k, v in d.items() if k not in ['next', 'prev']}

            ### The part that deal with node by differen conditions ###  
            merged_dll_node = DoublyListNode() # store the aggregated info from the dll_nodes
                 
            if 'module' in first_keys: # For the modules stored at the head of the dll, capsule it by nn.Sequential()
                values = [dll_node.module for dll_node in filtered_dll_nodes] # Extract the 'module' values from the nodes at the current position

                merged_dll_node.module = nn.Sequential(*values) # update merged_dll_node by the capsuled modules using nn.Sequential()
            elif 'depth' in first_keys:

                # Ensure all 'depth' values are the same before merging
                depths = [dll_node.depth for dll_node in filtered_dll_nodes]
                if len(set(depths)) != 1:
                    raise ValueError("All 'depth' values must be the same when merging modules")

                merged_dll_node.depth = depths[0] # update the depth 
            
            # If the attributes except next and prev are the same 
            elif all(filter_basic_attribute(dll_node.__dict__).values() == filter_basic_attribute(first_dll_node_attributes).values() for dll_node in filtered_dll_nodes):
                # If values are the same, take the first node's values as the value for the merged node
                merged_dll_node.__dict__.update(filter_basic_attribute(first_dll_node_attributes)) # update the attributes in the way of dictionary 

            else:
                # For other conflicting data, set to None
                merged_dll_node.__dict__.update({key: None for key in first_keys})

            merged_node.info.append(merged_dll_node) # append the merged_dll_node

        # Perform sync_traverse on the selected children to merge their DLLs
        self.tree.sync_traverse(parent_node, aggregate_action)

        # Remove the merged child nodes from the parent's list of children
        for child in children_to_merge:
            parent_node.children.remove(child)

        # Insert the new merged node into the parent's list of children at position `n`
        parent_node.children.insert(n, merged_node)
        
    def _remove_node(self, node):
        pass

    def _insert_node_after(self, node):
        pass
    
    def _replicate_info(self, node):
        pass

    def prune(self, depth):
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
tree_manager = PyTreeManager()
tree_manager.build_tree(model, depth=2)  # Build the tree up to depth 2
print(tree_manager)  # Print the tree structure
