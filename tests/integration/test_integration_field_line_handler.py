# -*- coding: utf-8 -*-
"""
Created on Tue June 22 2021

@author: lordofbejgli
"""

import unittest
import os
import numpy as np

from flap_field_lines.field_line_handler import *

class TestLoadingData(unittest.TestCase):
    """
    This is a series of tests the check the behaviour of FieldLineHandler's 
    constructor. The test data is not present in the repo, for test files are 
    massive. These tests are skipped if the path given for test data does not 
    exist. To run them, first obtain flux surface files for then'EIM' W7X 
    magnetic configuration and adjust the setUp() method to point to their 
    location.
    """
    #Give flux surface files location here
    data_path = '/media/data/w7x_flux_surfaces/test/fs_info.sav'

    def setUp(self) -> None:
        """
        Test initiation. Initializes a few parameters of the reading. Lines are 
        chose with multiple selection, while the toroidal coordinate selection 
        is given as a range.
        """
        self.lines = (5, 60, 120, 240)
        self.tor_range = '0:500:50'
        self.handler = FieldLineHandler(self.data_path, configuration='EIM')

    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_reading_all_lines(self):
        """
        Reading all lines from flux surface file.
        """
        self.handler.update_read_parameters(surfaces=30)
        self.handler.load_data()
        self.assertEqual(self.handler.return_field_lines().shape, (3, 360, 3651))
        self.handler.load_data(getB=True)
        self.assertEqual(self.handler.return_B().shape, (3, 360, 3651))
        self.assertEqual(self.handler.return_gradB(), None)
        self.handler.load_data(getGradB=True)
        self.assertEqual(self.handler.return_gradB().shape, (3, 360, 3651))
        self.assertEqual(self.handler.return_B().shape, (3, 360, 3651))

    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_read_directions(self):
        """
        These tests check the reading of field lines in various directions.
        """
        self.handler.update_read_parameters(surfaces=30, lines=self.lines,
                                            tor_range=self.tor_range, direction='backward')
        self.handler.load_data()
        self.assertEqual(self.handler.return_field_lines().shape, (3, 4, 10))
        self.handler.update_read_parameters(surfaces=30, lines=self.lines,
                                            tor_range=self.tor_range, direction='both')
        self.handler.load_data()
        self.assertEqual(self.handler.return_field_lines().shape, (3, 4, 10))
        field_lines = self.handler.return_field_lines()
        self.handler.update_read_parameters(surfaces=30, lines=self.lines,
                                            tor_range='-1:-500:-50', direction='backward')
        self.handler.load_data()
        self.assertTrue(np.array_equal(field_lines, self.handler.return_field_lines()))

    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_read_multiple_files(self):
        """
        Checks reading from multiple files. Also tests correct behaviour if 
        some files are not found. Checks if fs_info is properly read.
        """
        self.handler.update_read_parameters(lines=self.lines,
                                            tor_range=self.tor_range)
        self.handler.load_data()
        self.assertEqual(self.handler.return_field_lines().shape, (3, 4, 10, 2))

        self.handler.update_read_parameters(surfaces=(25, 30, 35), lines=self.lines,
                                            tor_range=self.tor_range)
        self.handler.load_data()
        self.assertEqual(self.handler.return_field_lines().shape, (3, 4, 10))

        self.handler.update_read_parameters(surfaces="30:41:5", lines=self.lines,
                                            tor_range=self.tor_range)
        self.handler.load_data()
        self.assertEqual(self.handler.return_field_lines().shape, (3, 4, 10, 2))

        fs_info = self.handler.return_fs_info()
        self.assertTrue(np.array_equal(fs_info['separatrix'], [83, 43, 55, 65, 76, 87]))

    @unittest.skipIf(not os.path.exists(data_path), "Skip if test data path is nonexistent.")
    def test_read_B_gradB(self):
        """
        Checks reading B and gradB.
        """
        self.handler.update_read_parameters(surfaces=(30, 40), lines=self.lines,
                                            tor_range=self.tor_range)
        self.handler.load_data(getB=True)
        self.assertEqual(self.handler.return_B().shape, (3, 4, 10, 2))
        
        self.handler.update_read_parameters(surfaces=(30, 40), lines=self.lines,
                                            tor_range=self.tor_range)
        self.handler.load_data(getGradB=True)
        self.assertEqual(self.handler.return_gradB().shape, (3, 4, 10, 2))
        
        self.handler.update_read_parameters(surfaces=(30, 40), lines=self.lines,
                                            tor_range=self.tor_range)
        self.handler.load_data(getB=True, getGradB=True)
        self.assertEqual(self.handler.return_B().shape, (3, 4, 10, 2))
        self.assertEqual(self.handler.return_gradB().shape, (3, 4, 10, 2))
