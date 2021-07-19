# -*- coding: utf-8 -*-
"""
Created on Tue June 4 2021

@author: lordofbejgli
"""

import unittest

from flap_field_lines.field_line_handler import *
from errors import *

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
    data_path = '/media/data/w7x_flux_surfaces/test/fs_info.sav'

    def test_exceptions_raised(self):
        """
        Checks if proper exceptions are raised for faulty inputs and 
        nonexistent paths.
        """
        self.assertRaises(WrongConfigurationError, FieldLineHandler, configuration='retek')
        self.assertRaises(FileNotFoundError, FieldLineHandler, path='retek')
        self.assertRaises(NoFsInfoError, FieldLineHandler, configuration='EIM')

    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_constructor(self):
        handler = FieldLineHandler(self.data_path)
        fs = handler.return_fs_info()
        self.assertEqual(fs['iota'].shape, (95,))

class TestFieldLineHandlerFunctions(unittest.TestCase):
    """
    TBD
    """
    data_path = '/media/data/w7x_flux_surfaces/test/fs_info.sav'

    def setUp(self) -> None:
        self.handler = FieldLineHandler(self.data_path, 'EIM')

    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_create_surf_file_list(self):
        surfs = process_selection('25:45')
        files, surfs = self.handler.create_surf_file_list(surfs)
        self.assertEqual(surfs, [30, 40])
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0], '/media/data/w7x_flux_surfaces/test/field_lines_tor_ang_1.85_1turn_EIM+252_w_o_limiters_w_o_torsion_w_characteristics_surf_030.sav')


if __name__ == '__main__':
    unittest.main(verbosity=2)
