import unittest
from nnlm.base import *

import unittest

class TestDoublyListNode(unittest.TestCase):

    def test_node_str_with_attributes(self):
        """Test if __str__ prints all attributes except next and prev correctly"""
        node = DoublyListNode()
        node.attribute1 = 'test1'
        node.attribute2 = 'test2'
        self.assertEqual(str(node), 'attribute1: test1, attribute2: test2')

    def test_node_str_with_no_attributes(self):
        """Test if __str__ returns 'None' when no attributes (besides next and prev) are present"""
        node = DoublyListNode()
        self.assertEqual(str(node), 'None')


class TestDoublyLinkedList(unittest.TestCase):

    def setUp(self):
        """Set up an empty doubly linked list before each test"""
        self.dll = DoublyLinkedList()

    def test_append(self):
        """Test appending elements to the doubly linked list"""
        node1 = DoublyListNode()
        node1.data = 'A'
        node2 = DoublyListNode()
        node2.data = 'B'

        self.dll.append(node1)
        self.dll.append(node2)

        self.assertEqual(str(self.dll), "data: A <-> data: B")  # Check list content
        self.assertEqual(self.dll.head, node1)   # Check head
        self.assertEqual(self.dll.tail, node2)   # Check tail

    def test_insert_after(self):
        """Test inserting an element after a given node"""
        node1 = DoublyListNode()
        node1.data = 'A'
        node2 = DoublyListNode()
        node2.data = 'B'
        node3 = DoublyListNode()
        node3.data = 'C'

        self.dll.append(node1)
        self.dll.append(node2)
        self.dll.insert_after(node1, node3)

        self.assertEqual(str(self.dll), "data: A <-> data: C <-> data: B")  # Check insertion result

    def test_delete(self):
        """Test deleting an element from the list"""
        node1 = DoublyListNode()
        node1.data = 'A'
        node2 = DoublyListNode()
        node2.data = 'B'
        node3 = DoublyListNode()
        node3.data = 'C'

        self.dll.append(node1)
        self.dll.append(node2)
        self.dll.append(node3)
        self.dll.delete(node2)  # Delete "B"

        self.assertEqual(str(self.dll), "data: A <-> data: C")  # Check deletion result
        self.dll.delete(node1)  # Delete "A"
        self.assertEqual(str(self.dll), "data: C")  # Check deletion result

    def test_traverse(self):
        """Test traversing the list"""
        node1 = DoublyListNode()
        node1.data = 'A'
        node2 = DoublyListNode()
        node2.data = 'B'
        node3 = DoublyListNode()
        node3.data = 'C'

        self.dll.append(node1)
        self.dll.append(node2)
        self.dll.append(node3)
        elements = []

        def collect_data(node):
            elements.append(node.data)

        self.dll.traverse(collect_data)
        self.assertEqual(elements, ['A', 'B', 'C'])  # Check traversal result

    def test_len(self):
        """Test the length method of the linked list"""
        node1 = DoublyListNode()
        node1.data = 'A'
        node2 = DoublyListNode()
        node2.data = 'B'

        self.dll.append(node1)
        self.dll.append(node2)
        self.assertEqual(len(self.dll), 2)  # Check the length of the list

    def test_str(self):
        """Test the __str__ method"""
        node1 = DoublyListNode()
        node1.data = 'A'
        node2 = DoublyListNode()
        node2.data = 'B'

        self.dll.append(node1)
        self.dll.append(node2)
        self.assertEqual(str(self.dll), "data: A <-> data: B")  # Check string representation


if __name__ == '__main__':
    unittest.main()

