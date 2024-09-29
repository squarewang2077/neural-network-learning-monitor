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

    def _expand_leaves(self):
        '''
        Traverse the tree and expand all the leaves having .children attribute in the head of the DLL node. Adding the children to the tree node and updating the depth of the children by increasing 1.
        '''
        def expansion_action(node):
            """
            Action function to expand the leaves of the tree.
            """
            # Check if the node is a leaf (i.e., has no children)
            if not node.children:
                # Safely check for children (submodules) in the current PyTorch module
                if hasattr(node.info.head.module, 'children'):
                    for child in node.info.head.module.children():
                        child_node = self.tree.add_child(node)  # Add child node to the current tree node

                        # Add info for the child node
                        child_dll_node = DoublyListNode()  # the dll node to store the information
                        child_dll_node.module = child
                        child_dll_node.depth = node.info.head.depth + 1

                        child_node.info.append(child_dll_node)  # append on the child_node.info

        # Use the traverse method to expand the leaves
        self.tree.traverse(self.tree.root, expansion_action)

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
                 
            if all(key in first_keys for key in ['module', 'depth']):  # For the modules stored at the head of the dll, capsule it by nn.Sequential()
                values = [dll_node.module for dll_node in filtered_dll_nodes] # Extract the 'module' values from the nodes at the current position

                # Ensure all 'depth' values are the same before merging
                depths = [dll_node.depth for dll_node in filtered_dll_nodes]
                if len(set(depths)) != 1:
                    raise ValueError("All 'depth' values must be the same when merging modules")

                merged_dll_node.module = nn.Sequential(*values) # update merged_dll_node by the capsuled modules using nn.Sequential()

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

    def merge_nodes(self, parent_child_nodes_dict):
        '''
        This method warp the _merge_nodes method to merge the adjacent sibling nodes from the parent_child_nodes_dict. After the merge, the leaves will be expanded.
        Args: 
            parent_child_nodes_dict: the dictionary that contains the parent node and the child nodes to be merged. The parent node is the key and the value is a list of the child nodes to be merged. 
        '''

        def pop_reindex(index_pair_list):
            '''
            This function is to pop the first pair of the index_pair_list and reindex the rest of the index_pair_list by subtracting the length of the first pair.
            Return:
                first_pair: the first pair of the index_pair_list
            '''
            first_pair = index_pair_list.pop(0)
            first_pair_length = first_pair[1] - first_pair[0]

            for i in range(len(index_pair_list)):
                index_pair_list[i] = (index_pair_list[i][0] - first_pair_length, index_pair_list[i][1] - first_pair_length)

            return first_pair[0], first_pair[1]

        for parent_node, index_pair_list in parent_child_nodes_dict.items():
            '''
                go through the parent_child_nodes_dict and merge the adjacent sibling nodes from the parent_node    
            '''
            while index_pair_list: # merge all the adjacent sibling nodes from the parent_node
                m, n = pop_reindex(index_pair_list)
                self._merge_nodes(parent_node, m, n)

        self._expand_leaves() # expand the leaves after merging the nodes
        self._alias('add')

    def _find_parent_node(self, target_node: TreeNode):

        return self.tree._find_parent_node(target_node)

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

    def _remove_node(self, node_to_remove: TreeNode):
        """
        Remove a node from the tree without removing its children.
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
    
    def prune(self, retained_depth):
        """
        Prunes the tree by marking all nodes except the root, the leaves, and the nodes at the specified depths for removal. Then, removes all the marked nodes.
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

        # filter out the leaf nodes 
        nonleaf_nodes = [node for node in nodes_to_remove if node.children]
        nodes_to_remove = nonleaf_nodes

        # Now, remove the marked nodes
        for node in nodes_to_remove: 
            self._remove_node(node)

        self._alias("remove")
        self._alias("add")

    def _alias(self, add_or_remove="add"): 
        """
        Assigns or removes a unique name to/from the head node of each DLL in the tree.
        The name is the name of the module and the depth of the module in a short form.
        This name is created as an attribute of the DLL node, which is .alias.
        
        Args:
            add_or_remove (str): If "add", adds the alias. If "remove", removes the alias.
        """
        def rename_action(node):
            def dll_rename_action(dll_node):
                if hasattr(dll_node, 'module') and hasattr(dll_node, 'depth'):
                    if add_or_remove == "add":
                        module_name = dll_node.module.__class__.__name__
                        depth = dll_node.depth
                        dll_node.alias = f"{depth}:{module_name}"
                    elif add_or_remove == "remove" and hasattr(dll_node, 'alias'):
                        del dll_node.alias

            node.info.traverse(dll_rename_action)

        self.tree.traverse(self.tree.root, rename_action)

    def get_max_depth(self):
        """
        Returns the maximum depth of the tree.
        """
        max_depth = 0

        def max_depth_action(node):
            nonlocal max_depth
            if node.info.head.depth > max_depth:
                max_depth = node.info.head.depth

        self.tree.traverse(self.tree.root, max_depth_action)

        return max_depth

    def _leaf_check(self):
        '''
        Traverse the tree nodes and check whether all the leaf nodes have the same depth.
        '''
        leaf_depths = set()

        def check_leaf_depth(node):
            # If the node has no children, it's a leaf node
            if not node.children:
                leaf_depths.add(node.info.head.depth)

        self.tree.traverse(self.tree.root, check_leaf_depth)

        # Check if all leaf nodes have the same depth
        if len(leaf_depths) == 1:
            return True
        else:
            return False

    def _get_attr(self, tree_node, attr_name):
        """
        Retrieves the attribute value of the DLL nodes in the tree.
        Args:
            tree_node: the node of the tree to retrieve the attribute
            attr_name: the attribute name to retrieve
        Returns:
            attr_values: the node and the corresponding attribute value
        """
        attr_values = []
        attr_nodes = []
    
        def get_attr_action(dll_node):
            if hasattr(dll_node, attr_name):
                attr_nodes.append(dll_node)
                attr_values.append(getattr(dll_node, attr_name))

        tree_node.info.traverse(get_attr_action)

        # to insure that the attribute has the same value or dll node in the tree nodes 
        assert (len(attr_values) <= 1) or (len(attr_nodes) <= 1), f"The attribute {attr_name} has different values or dll nodes in the tree nodes."

        # if no attribute value is found, return None
        if (len(attr_values) == 0) and (len(attr_nodes) == 0):
            return None, None
        # if the attribute value is None
        elif (len(attr_values) == 0) and (len(attr_nodes) > 0):
            return attr_nodes[0], None
        else:
            return attr_nodes[0], attr_values[0]

    def feature_tracker(self, batched_inputs):
        '''
        The function to track the features after each module in the module and store the features in the first empty DLL node, which is the second one. 
        '''
        if not self._leaf_check(): # check whether all the leaf nodes have the same depth
            raise ValueError("The leaf of the tree has more than one depth, which is not suitable for tracking features.")

        depth_tracker = -1 # insure that the depth will be changed at the first time
        output_tracker = {} # track the output features of the previous module by the diferent depth

        def calculate_output_features(tree_node):
            '''
            As we go through each node of the tree, there will be depth change tracker that if the depth is changed, the input features will be reassigned. 
            '''
            nonlocal depth_tracker
            nonlocal output_tracker
            current_depth = tree_node.info.head.depth # the current depth of the tree node
            if (current_depth != depth_tracker) and (current_depth not in output_tracker.keys()): # if the depth is changed and first tracked by the output_tracker
                batched_input_features = batched_inputs # reset the batched_input_features
            else: # if the depth is not changed
                batched_input_features = output_tracker[current_depth] # pass the output features of previous tree node to the current node

            depth_tracker = current_depth # update the depth_tracker

            # For each depth of the tree, compute the output features
            try: # catch the exception when the module is not able to compute the output features
                batched_output_features = tree_node.info.head.module(batched_input_features)
            except Exception as e:
                raise RuntimeError(f"Failed to compute output features for module {tree_node.info.head.module}: {e}")

            output_tracker[current_depth] = batched_output_features # update the output_tracker for the current node

            info_dll_node_in, info_dll_value_in =self._get_attr(tree_node, 'batched_input_features')

            info_dll_node_out, info_dll_value_out =self._get_attr(tree_node, 'batched_output_features')

            if info_dll_node_in and info_dll_node_out: # if the input and output features are found in the tree node
                info_dll_node_in.batched_input_features = batched_input_features
                info_dll_node_out.batched_output_features = batched_output_features
            elif (not info_dll_node_in) and (not info_dll_node_out):
                # Initilize the DLL node to store the computed intermediate features
                info_dll_node = DoublyListNode()

                # Append the input and output features to the tail DLL node of the current tree node 
                info_dll_node.batched_input_features = batched_input_features
                info_dll_node.batched_output_features = batched_output_features

                # Append the info_dll_node to the tree_node
                tree_node.info.append(info_dll_node)
            else: 
                raise AttributeError("The input and output features should co-exist.")

        self.tree.traverse(self.tree.root, calculate_output_features)

        # Consistency check for the output features
        
        assert sum([(v != output_tracker[0]).sum().item() for v in output_tracker.values()]) == 0, "Inconsistent output features for different depths in the tree."

    def update_info(self, func):
        """
        Updates the additional information to the DLL nodes, using traverse method.
        This is a compounded method based on other methods.
        Args:
            func: the function to update the information of the DLL node, the output of the func should be a dictionary 
            batched_inputs: the input data to the network passed to the feature_tracker
        """
        
        def tree_action(tree_node):
            dll_node = DoublyListNode() # initialize the DLL node to store the information
            info_dict = func(tree_node) # get the information from the function
            dll_node.__dict__.update(info_dict) # update the information to the DLL node
            tree_node.info.append(dll_node) # append the DLL node to the tree node

        self.tree.traverse(self.tree.root, tree_action)

    def search_info(self, conditions, keys):
        """
        Outputs the information of the DLL nodes in the tree that meet specified conditions.

        Args:
            conditions (dict): A dictionary of conditions to filter the DLL nodes.
            keys (list): A list of keys to extract information from the DLL nodes.

        Returns:
            dict: A dictionary containing the extracted information for each key.
        """
        output_dict = {key: [] for key in keys}  # Initialize the output dictionary

        def treenode_action(tree_node):
            def fetch_info(dll_node):
                # update the output_dict with the specified keys
                for key in keys:
                    if hasattr(dll_node, key):
                        output_dict[key].append(getattr(dll_node, key))

            # Check if the dll_node satisfies all conditions
            if all(any(getattr(dll_node, cond_key, None) == cond_value for dll_node in tree_node.info) for cond_key, cond_value in conditions.items()):
                tree_node.info.traverse(fetch_info)

        try:
            self.tree.traverse(self.tree.root, treenode_action)
        except AttributeError as e:
            raise RuntimeError(f"Failed to traverse the tree: {e}")

        return output_dict


    def __str__(self):
        """String representation of the tree manager."""
        return str(self.tree) if self.tree is not None else "No tree has been built."


