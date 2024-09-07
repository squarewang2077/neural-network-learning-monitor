import unittest
from nnlm.base import *


class TestDoublyLinkedList(unittest.TestCase):

    def setUp(self):
        """Set up an empty doubly linked list before each test"""
        self.dll = DoublyLinkedList()

    def test_append(self):
        """Test appending elements to the doubly linked list"""
        self.dll.append("A")
        self.dll.append("B")
        self.assertEqual(str(self.dll), "A <-> B")  # Check list content
        self.assertEqual(str(self.dll.head), "A")   # Check head
        self.assertEqual(str(self.dll.tail), "B")   # Check tail

    def test_insert_after(self):
        """Test inserting an element after a given node"""
        self.dll.append("A")
        self.dll.append("B")
        self.dll.insert_after(self.dll.head, "C")
        self.assertEqual(str(self.dll), "A <-> C <-> B")  # Check insertion result

    def test_delete(self):
        """Test deleting an element from the list"""
        self.dll.append("A")
        self.dll.append("B")
        self.dll.append("C")
        self.dll.delete(self.dll.head.next)  # Delete "B"
        self.assertEqual(str(self.dll), "A <-> C")  # Check deletion result
        self.dll.delete(self.dll.head)  # Delete "A"
        self.assertEqual(str(self.dll), "C")  # Check deletion result

    def test_traverse(self):
        """Test traversing the list"""
        self.dll.append("A")
        self.dll.append("B")
        self.dll.append("C")
        elements = []

        def collect_data(node):
            elements.append(str(node))

        self.dll.traverse(collect_data)
        self.assertEqual(elements, ["A", "B", "C"])  # Check traversal result

    def test_str(self):
        """Test the __str__ method"""
        self.dll.append("A")
        self.dll.append("B")
        self.assertEqual(str(self.dll), "A <-> B")  # Check string representation

if __name__ == '__main__':
    unittest.main()
