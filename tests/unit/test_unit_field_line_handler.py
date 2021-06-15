# -*- coding: utf-8 -*-
"""
Created on Tue June 4 2021

@author: lordofbejgli
"""

import unittest
import numpy as np

from flap_field_lines.field_line_handler import *

class TestAccessories(unittest.TestCase):
    """
    Tests for accessory functions of FieldLineHandler class.
    """
    def test_process_selection(self):
        """
        This test checks if proper output is given by process_selection for 
        proper input, and proper errors are raised if not.
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