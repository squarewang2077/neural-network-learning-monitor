from nnlm.base import *
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
            if depth is not None and node.info.head.depth >= depth: 
                return

            # Safely check for children (submodules) in the current PyTorch module
            try:
                for child in node.info.head.module.children():
            
                    child_node = self.tree.add_child(node) # Add child node to the current tree node

                    # Add info for the child node 
                    child_dll_node = DoublyListNode() # the dll node to store the infomation 
                    child_dll_node.module = child 
                    child_dll_node.depth = node.info.head.depth + 1

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
        # Ensure the self.tree is not None 
        assert self.tree is not None, "The root of the tree is None"
        
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

    def _find_parent_node(self, target_node: TreeNode):
        """
        Finds the parent of the given target node in the tree.

        Args:
            target_node (TreeNode): The node whose parent needs to be found.

        Returns:
            TreeNode: The parent node of the target node, or None if the target node is not found in the tree.
        """
        parent_node = None

        def search_for_parent(current_node):
            nonlocal parent_node
            if target_node in current_node.children:
                parent_node = current_node

        # Traverse the tree to find the parent of the target_node
        self.tree.traverse(self.tree.root, search_for_parent)

        return parent_node


    def _remove_node(self, node_to_remove: TreeNode):
        """
        Remove a node from the tree without removing its children.
        If the removed node is the root, all its children will become new independent trees.
        """
        # Error Checking about the root 
        assert self.tree is not None, "The root of the tree is None"
        assert node_to_remove != self.tree.root, "Cannot remove the root node"  

        # Traverse the tree to find the parent of the node to remove
        ## We have to do this because the tree node only have .child not .parent
        parent_node = self._find_parent_node(node_to_remove)

        # If no parent was found, the node is not part of the tree
        if parent_node is None:
            raise ValueError("Node not found in the tree.")

        # Insert the children of node_to_remove into its parent's list of children
        index = parent_node.children.index(node_to_remove)
        parent_node.children.pop(index)  # Remove the node_to_remove from its parent's children

        def update_depth(node):
            """
            Update the depth of each child node of node_to_remove by decreasing 1.
            """
            # this is another way to use the traverse defined, it can be a iterate, by defining the
            def dll_action(dll_node):
                if hasattr(dll_node, 'depth'):  # Ensure the node has a depth attribute
                    dll_node.depth -= 1  # Increment the depth by 1
            
            node.info.traverse(dll_action)

        # Apply the update_depth to each child of the node_to_remove
        for child in node_to_remove.children:
            self.tree.traverse(child, update_depth)  # Traverse each subtree rooted at the children

        # Insert the children of the node to remove in its place in reverse order to preserve order
        for child in reversed(node_to_remove.children): 
            parent_node.children.insert(index, child)


    def _replicate_info(self, node: TreeNode):
        """
        Replicates all DLL nodes from the given tree node.
        All attributes except 'next' and 'prev' are set to None.

        Args:
            node (TreeNode): The tree node from which to replicate DLL nodes.
        Returns:
            node (TreeNode): The new tree node with replicated information   
        """
        # Create a new node to hold the replicated information
        replicated_node = TreeNode() 

        def replicate_dll_info(dll_node):
            """
            Action to replicate a DLL node, setting all attributes (except 'next' and 'prev') to None.
            """
            # Create a new DLL node for replication
            new_dll_node = DoublyListNode()

            # Set all attributes to None, except 'next' and 'prev'
            for attr in dll_node.__dict__.keys():
                if attr not in ['next', 'prev']:  # Only preserve 'next' and 'prev'
                    setattr(new_dll_node, attr, None)

            # Append the replicated DLL node to the new node's doubly linked list
            replicated_node.info.append(new_dll_node) # the nonlocal is not need, since replicated_node is global variable  

        # Traverse through the DLL nodes of the original node and replicate them
        node.info.traverse(replicate_dll_info)

        return replicated_node

    def _insert_node_after(self, node: TreeNode):
        """
        Inserts a new node after the given node, where the new node's information
        is created by replicating the information of the given node.
        
        Args:
            node (TreeNode): The node after which the new node will be inserted.
        """
        # Ensure the tree exists
        assert self.tree is not None, "The tree is not initialized."
        
        # Create the new node by replicating the info of the given node
        new_node = self._replicate_info(node)
        
        # Find the parent of the given node
        parent_node = self._find_parent_node(node)
        
        # If no parent is found, the node is not part of the tree
        if parent_node is None:
            raise ValueError("Node not found in the tree.")
        
        # Get the index of the given node in the parent's children list
        index = parent_node.children.index(node)
        
        # Insert the new node after the given node in the parent's children list
        parent_node.children.insert(index + 1, new_node)

    
    def prune(self, retained_depth):
        """
        Prunes the tree by marking all nodes except the root and the nodes at the specified depths for removal.
        Then, removes all the marked nodes.

        Args:
            retained_depth (list): A list of integer depths to retain in the tree. 
                                Nodes at these depths will not be removed.
        """
        
        assert self.tree is not None, "The tree is not initialized."

        # First, mark the nodes that need to be removed
        nodes_to_remove = []

        # Helper function to mark nodes during traversal
        def prune_action(node):
            def dll_prune_action(dll_node):
                if hasattr(dll_node, 'depth'):
                    node_depth = dll_node.depth
                    # Mark the node if its depth is not in the retained_depth
                    if node_depth not in retained_depth and node_depth != 0:  # Always keep the root node (depth = 0)
                        # Mark the node for removal
                        nodes_to_remove.append(node)
            
            node.info.traverse(dll_prune_action)

        # Traverse the tree and mark nodes for removal
        self.tree.traverse(self.tree.root, prune_action)

        # Now, remove the marked nodes
        for node in nodes_to_remove:
            self._remove_node(node)

    def __str__(self):
        """String representation of the tree manager."""
        return str(self.tree) if self.tree is not None else "No tree has been built."


