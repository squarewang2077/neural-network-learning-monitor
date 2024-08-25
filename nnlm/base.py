class DoublyListNode:
    def __init__(self, data):
        self.data = data  # Can be any data type
        self.next = None
        self.prev = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, data):
        new_node = DoublyListNode(data)
        if self.head is not None: # if the linkedlist is not empty
            # establish the connection 
            self.tail.next = new_node 
            new_node.prev = self.tail
            # move the tail pointer
            self.tail = new_node
        else: # if the linkedlist is empty
            self.head = new_node
            self.tail = new_node

    def insert_after(self, node, data):
        new_node = DoublyListNode(data)
        # establish the connection from the new node
        new_node.prev = node
        new_node.next = node.next
        # establish the connection from prev and next nodes 
        node.next = new_node
        if node.next is not None: # if the node is not tail 
            node.next.prev = new_node
        else: # if the node is the tail node
            assert node == self.tail, "DoublyLinkedList->insert_after->The node needs to be the tail node"
            # move the tail pointer
            self.tail = new_node

    def delete(self, node):
        # short cut from the prev node
        if node.prev is not None: # if the node is not head
            # short cut the connect 
            node.prev.next = node.next
        else: # if the node is head
            assert node == self.head, "DoublyLinkedList->delete->The node needs to be the head node"
            # move the head pointer 
            self.head = node.next
            if self.head is not None: # if the delete node is not the only node
                self.head.prev = None # reset the head.prev to None

        # short cut to the next
        if node.next is not None: # if the node is not the tail 
            node.next.prev = node.prev # short cut the next connection
        else: # if the node is tail 
            assert node == self.tail, "DoublyLinkedList->delete->The node needs to be the tail node"
            # move the tail pointer
            self.tail = node.prev
            if self.tail is not None: # if the node is not the only one
                self.tail.next = None # reset the tail.next to None

    def traverse_forward(self):
        current = self.head
        while current:
            print(current.data)
            current = current.next

    def traverse_backward(self):
        current = self.tail
        while current:
            print(current.data)
            current = current.prev

    def __str__(self):
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return " <-> ".join(elements)

class TreeNode:
    """The node in the tree"""
    def __init__(self):
        self.info = DoublyLinkedList()  # Linked list to store monitoring information
        self.children = []

    def __repr__(self):
        return f"TreeNode()"

class Tree:
    """The basic tree structure"""
    def __init__(self, tag: str):
        self.root = TreeNode()
        self.tag = tag  # Tag to distinguish between different trees

    def add_child(self, parent_node):
        """Adds a child node to a specified parent node as the last child"""
        child_node = TreeNode()
        parent_node.children.append(child_node)
        return child_node

    def delete_child(self, parent_node, child_node):
        """Deletes a child node from its parent"""
        parent_node.children.remove(child_node)

    def traverse(self, node=None, level=0):
        """Traverses the tree and prints the structure"""
        if node is None:
            node = self.root
        indent = " " * (level * 4)
        print(f"{indent}{node.info}")
        for child in node.children:
            self.traverse(child, level + 1)

class TreeManager:
    """Manages multiple trees, module names, and handles advanced operations"""
    def __init__(self):
        self.main_tree = None
        self.monitoring_trees = []
        self.module_names = {}  # Manages module names for each TreeNode
        self.registered_nodes = {}  # Stores aliases and access names for nodes

    def build_tree(self, root_module_name, tree_id: int, get_children_fn, tag: str, depth=None):
        """Initializes and builds the main tree."""
        self.main_tree = Tree(tag)
        self._build_tree_recursive(self.main_tree.root, root_module_name, get_children_fn, tree_id, 1, depth)
        self.set_module_name(self.main_tree.root, root_module_name)

    def _build_tree_recursive(self, current_node, module, get_children_fn, tree_id, level, depth):
        """Helper function to recursively build a tree with optional depth limit."""
        if depth is not None and level > depth:
            return
        children = get_children_fn(module)
        position = 0
        for child in children:
            child_node = self.main_tree.add_child(current_node)
            self.set_module_name(child_node, str(child))
            self._build_tree_recursive(child_node, child, get_children_fn, tree_id, level + 1, depth)
            position += 1

    def get_module_name(self, node):
        """Retrieves the module name for a given TreeNode."""
        return self.module_names.get(node, "")

    def set_module_name(self, node, name):
        """Sets the module name for a given TreeNode."""
        self.module_names[node] = name

    def merge_siblings(self, parent_node, start_idx, end_idx):
        """Merge siblings from start_idx to end_idx under the same parent node."""
        nodes_to_merge = parent_node.children[start_idx:end_idx + 1]
        merged_node = TreeNode()
        for node in nodes_to_merge:
            merged_node.children.extend(node.children)
            parent_node.children.remove(node)
        parent_node.children.insert(start_idx, merged_node)

    def merge(self, nodes):
        """Merge nodes from different parents at the same depth."""
        if not nodes:
            return
        merged_node = TreeNode()
        for node in nodes:
            merged_node.children.extend(node.children)
        parent_node = nodes[0].parent  # Assume nodes share the same parent
        parent_node.children = [child for child in parent_node.children if child not in nodes]
        parent_node.children.append(merged_node)

    def split_node(self, parent_node, sub_node_position):
        """Split a node into two parts based on sub_node_position."""
        node_to_split = parent_node.children[sub_node_position]
        left_child = TreeNode()
        right_child = TreeNode()
        midpoint = len(node_to_split.children) // 2
        left_child.children = node_to_split.children[:midpoint]
        right_child.children = node_to_split.children[midpoint:]
        parent_node.children[sub_node_position] = left_child
        parent_node.children.insert(sub_node_position + 1, right_child)

    def insert_after(self, node):
        """Insert a new node after the given node."""
        parent_node = node.parent
        new_node = TreeNode()
        parent_node.children.insert(parent_node.children.index(node) + 1, new_node)
        return new_node

    def delete(self, node):
        """Delete a given node from the tree."""
        parent_node = node.parent
        parent_node.children.remove(node)

    def register(self, node, alias=None):
        """Register a node with an alias for easier access."""
        if alias:
            self.registered_nodes[alias] = node
        else:
            identifier = f"Tree_{id(node)}_Level_{len(self.registered_nodes)}"
            self.registered_nodes[identifier] = node

    def access(self, alias):
        """Access a node by its alias."""
        return self.registered_nodes.get(alias)

    def update_info(self, nodes, func):
        """Apply a function to update information on the given nodes."""
        for node in nodes:
            func(node)

    def access_data(self, nodes):
        """Access data stored in the monitored nodes."""
        return [node.info for node in nodes]

    def show_structure(self):
        """Show the structure of the main tree."""
        self.main_tree.traverse()

    def consistency_check(self):
        """Check if the tree structure is consistent with the original model."""
        # Implement consistency check logic here
        pass

    def sync_trees(self):
        """Synchronizes all monitoring trees with the main tree."""
        for monitoring_tree in self.monitoring_trees:
            self._sync_recursive(self.main_tree.root, monitoring_tree.root)

    def _sync_recursive(self, main_node, monitoring_node):
        """Recursively synchronize two trees."""
        self.set_module_name(monitoring_node, self.get_module_name(main_node))
        monitoring_node.children = main_node.children  # Assuming a shallow copy; otherwise, handle copying
        for main_child, monitoring_child in zip(main_node.children, monitoring_node.children):
            self._sync_recursive(main_child, monitoring_child)

    def print_identifiers(self):
        """Prints the identifiers for all nodes in all trees."""
        print("Node Identifiers:")
        for alias, node in self.registered_nodes.items():
            print(f"{alias}: {self.get_module_name(node)}")

# Example usage:
import torch.nn as nn
import torch.nn 

class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def torch_children_getter(module):
    """Function to get children of a PyTorch module."""
    return list(module.children())

# Initialize the TreeManager
nntreemanager = TreeManager()

# Build the main tree
model = ExampleModel()
nntreemanager.build_tree(root_module_name=str(model), tree_id=1, get_children_fn=torch_children_getter, tag='main', depth=None)

# Show the structure of the tree
nntreemanager.show_structure()

# Register and access a node
nntreemanager.register(nntreemanager.main_tree.root, alias="root_node")
accessed_node = nntreemanager.access("root_node")

# Update information on nodes using a function
nntreemanager.update_info([nntreemanager.main_tree.root], lambda node: node.info.append("Updated"))

# Access data from nodes
data = nntreemanager.access_data([nntreemanager.main_tree.root])
print("Accessed data:", data)

# Perform operations like merging and splitting
nntreemanager.merge_siblings(nntreemanager.main_tree.root, 0, 1)
nntreemanager.split_node(nntreemanager.main_tree.root, 0)

# Sync trees
nntreemanager.sync_trees()

# Traverse and display the trees again after modification
print("\nAfter modifications:")
nntreemanager.show_structure()

# Perform a consistency check
nntreemanager.consistency_check()

# Print identifiers
nntreemanager.print_identifiers()
