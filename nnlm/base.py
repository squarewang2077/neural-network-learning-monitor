class DoublyListNode:
    def __init__(self, data):
        self.data = data  # Can be any data type
        self.next = None
        self.prev = None

    def __str__(self) -> str:
        return str(self.data) # the print of self.data depends on the __str__ method of the data


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
    def traverse(self, action, backward=False):
        '''
            This method will traverse the linkedlist forward or backward,
            and put some action on each of the traversed nodes 
        '''
        assert backward in [True, False], "The parameter backward has to be True or False"
        
        if not backward: 
            current = self.head
            while current is not None:
                action(current)  
                current = current.next
        elif backward:
            current = self.tail
            while current is not None:
                action(current)
                current = current.prev

    def __str__(self):
        '''
            Print out the entire list of all elements using the traverse method
        '''
        
        result = [] # Define a list to collect node data during traversal

        
        def collect_data(node): # Function to append each node's data to the result list
            result.append(str(node))

        
        self.traverse(collect_data) # Traverse the list and collect all node data

        
        return " <-> ".join(result) if result else "Empty List" # Join the collected data into a single string, separated by arrows
    

class TreeNode:
    """The node in the tree"""
    def __init__(self):
        self.info = DoublyLinkedList()  # Linked list to store monitoring information
        self.children = []

    def __str__(self) -> str:
        '''
            Since the tree node store the DoublyLinkedList, 
            it only print out the first element of the Linkedlist 
        '''
        return str(self.info.head) if self.info.head else "Empty"

class Tree:
    """The basic tree structure"""
    def __init__(self, tag: str):
        self.root = TreeNode()
        self.tag = tag  # Tag to distinguish between different trees

    def add_child(self, parent_node):
        """Adds a child node to a specified parent node as the last child"""
        child_node = TreeNode()
        parent_node.children.append(child_node) # [].append
        return child_node

    def delete_child(self, parent_node, child_node):
        """ 
        Deletes a child node from its parent
        This is the simple method without unique identification of each node
        The more advanced method only need the identification for the node 
        """
        parent_node.children.remove(child_node) # [].remove

    def traverse(self, node=None, action=None):
        """
        Performs pre-order traversal on the tree, applying an action (function) to each node.
        """
        if node is None:  # Start at the root if no node is provided
            node = self.root
        
        action(node) # Apply the action to the current node

        for child in node.children: # Recursively traverse each child node
            self.traverse(child, action)

    def __str__(self):
        """
        Prints out the entire tree structure in a tree-like format.
        """
        result = []
        depth = {self.root: 0}  # Dictionary to track the depth of each node

        # Define a helper function that only receives the node as an argument
        def collect_node_data(node):
            # Get the depth of the current node
            current_depth = depth[node]
            
            # Generate the tree-like prefix based on the depth
            if current_depth == 0:
                prefix = ""
            else:
                prefix = "│   " * (current_depth - 1) + "├── "

            # Append the formatted node data to the result
            result.append(f"{prefix}{str(node)}")

            # Update the depth for the children of this node
            for child in node.children:
                depth[child] = current_depth + 1

        # Use the traverse method to collect all nodes, passing only the node to the function
        self.traverse(action=collect_node_data)

        # Join all lines and return as a single string
        return "\n".join(result)
    
