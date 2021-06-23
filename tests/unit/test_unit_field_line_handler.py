# -*- coding: utf-8 -*-
"""
Created on Tue June 4 2021

@author: lordofbejgli
"""

import unittest

from flap_field_lines.field_line_handler import *

class TestAccessories(unittest.TestCase):
    """
    Tests for accessory functions of FieldLineHandler class.
    """
    def test_process_selection(self):
        """
        This test checks if proper output is given by process_selection for 
        proper input, and proper errors are raised if not.
        Checks for single input, range and individual selection.
        Checks for proper exceptions in the case of non-int input and 
        improper string format.
        """
        self.assertEquals([5], process_selection(5))
        self.assertEqual(range(3, 7), process_selection('3:7'))
        self.assertEqual(range(3, 20, 5), process_selection('3:20:5'))
        self.assertListEqual([3, 8, 13], process_selection([3, 8, 13]))
        self.assertListEqual([3, 8, 13], process_selection((3, 8, 13)))
        self.assertRaises(TypeError, process_selection)
        self.assertRaises(TypeError, process_selection, 3.75)
        self.assertRaises(TypeError, process_selection, (1, 3.2))
        self.assertRaises(ValueError, process_selection, 'retek')

class TestFieldLineHandlerConstructor(unittest.TestCase):
    """
    These tests check the FieldLineHandler constructor with various correct or 
    incorrect inputs.
    """
    def test_exceptions_raised(self):
        """
        Checks if proper exceptions are raised for faulty inputs and 
        nonexistent paths.
        """
        self.assertRaises(ValueError, FieldLineHandler, configuration='retek')
        self.assertRaises(ValueError, FieldLineHandler, direction='retek')
        self.assertRaises(IOError, FieldLineHandler, path='retek')


if __name__ == '__main__':
    unittest.main(verbosity=2)
