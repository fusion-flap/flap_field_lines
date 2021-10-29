# -*- coding: utf-8 -*-
"""
Created on Tue June 4 2021

@author: lordofbejgli
"""

import unittest

from flap_field_lines.field_line_handler import *
from flap_field_lines.errors import *

try:
    from ..config import data_path
except ImportError:
    from ..config_default import data_path  

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
        self.assertEqual([5], process_selection(5))
        self.assertEqual(range(3, 7), process_selection('3:7'))
        self.assertEqual(range(3, 20, 5), process_selection('3:20:5'))
        self.assertEqual(None, process_selection(':'))
        self.assertListEqual([3, 8, 13], process_selection([3, 8, 13]))
        self.assertListEqual([3, 8, 13], process_selection((3, 8, 13)))
        self.assertRaises(TypeError, process_selection)
        self.assertRaises(TypeError, process_selection, 3.75)
        self.assertRaises(TypeError, process_selection, (1, 3.2))
        self.assertRaises(ValueError, process_selection, 'retek')
        self.assertRaises(ValueError, process_selection, '5')
        self.assertRaises(ValueError, process_selection, '5:10:2:3')

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
        self.assertRaises(WrongConfigurationError, FieldLineHandler, configuration='retek')
        self.assertRaises(FileNotFoundError, FieldLineHandler, path='retek')
        self.assertRaises(NoFsInfoError, FieldLineHandler, configuration='EJM')

    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_constructor(self):
        handler = FieldLineHandler(data_path)
        fs = handler.return_fs_info()
        self.assertEqual(fs['iota'].shape, (95,))
        self.assertTrue(os.path.exists(handler.path))

class TestFieldLineHandlerFunctions(unittest.TestCase):
    """
    TBD
    """

    def setUp(self) -> None:
        self.handler = FieldLineHandler(data_path, 'EIM')

    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_create_surf_file_list(self):
        surfs = process_selection((30, 40, 96))
        files, surfs = self.handler.create_surf_file_list(surfs)
        self.assertEqual(surfs, [30, 40])
        self.assertEqual(len(files), 2)
        self.assertTrue(os.path.isfile(files[0]))

    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_create_surf_list(self):
        files = ['/media/data/w7x_flux_surfaces/test/field_lines_tor_ang_1.85_1turn_EIM+252_w_o_limiters_w_o_torsion_w_characteristics_surf_030.sav',
                 '/media/data/w7x_flux_surfaces/test/field_lines_tor_ang_1.85_1turn_EIM+252_w_o_limiters_w_o_torsion_w_characteristics_surf_040.sav']
        surfs = self.handler.create_surf_list(files)
        self.assertEqual([30, 40], surfs)

    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_update_read_parameters_errors(self):
        self.assertRaises(WrongDirectionError, self.handler.update_read_parameters, direction='retek')
        self.assertRaises(ValueError, self.handler.update_read_parameters, surfaces='retek')
        self.assertRaises(ValueError, self.handler.update_read_parameters, surfaces='3:4:12:b')
        
    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_update_read_parameters_file_selection(self):    
        expected_files = [self.handler.path + '/field_lines_tor_ang_1.85_1turn_EIM+252_w_o_limiters_w_o_torsion_w_characteristics_surf_030.sav',
                          self.handler.path + '/field_lines_tor_ang_1.85_1turn_EIM+252_w_o_limiters_w_o_torsion_w_characteristics_surf_040.sav'] 
        self.handler.update_read_parameters()
        self.assertGreater(len(self.handler.return_surface_files()), 0)
        self.handler.update_read_parameters(surfaces=[30, 40])
        self.assertEqual(self.handler.return_surface_files(), expected_files)
        self.handler.update_read_parameters(surfaces=expected_files)
        self.assertEqual(self.handler.return_surfaces(), [30, 40])
        self.handler.update_read_parameters()
        self.assertGreater(len(self.handler.return_surface_files()), 0)
        self.handler.update_read_parameters(surfaces=[30])
        self.assertEqual(len(self.handler.return_surface_files()), 1)
        self.handler.update_read_parameters(surfaces=40, drop_data=False)
        self.assertEqual(len(self.handler.return_surface_files()), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
