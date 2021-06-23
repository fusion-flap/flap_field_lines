# -*- coding: utf-8 -*-
"""
Created on Tue June 22 2021

@author: lordofbejgli
"""

import unittest
import numpy as np

from flap_field_lines.field_line_handler import *

class TestConstructor(unittest.TestCase):
    """
    This is a series of tests the check the behaviour of FieldLineHandler's 
    constructor. The test data is not present in the repo, for test files are 
    massive. In the main repo these tests either wil not be present or will be 
    otherwise deactivated. To run them, first obtain flux surface files for the
    'EIM' W7X magnetic configuration and adjust the setUp() method to point to 
    their location.
    """
    def setUp(self) -> None:
        """
        Test initiation. Initializes path for test data and a few parameters of 
        the reading. Lines are chose with multiple selection, while the toroidal 
        coordinate selection is given as a range.
        """
        #Give flux surface files location here
        self.path = '/media/data/w7x_flux_surfaces/test'
        self.lines = (5, 60, 120, 240)
        self.tor_range = '0:500:50'

    def test_reading_all_lines(self):
        """
        Reading all lines from flux surface file.
        """
        field_lines = FieldLineHandler(self.path, surface=40)
        self.assertEqual(field_lines.get_field_lines().shape, (3, 360, 3651))

    def test_read_directions(self):
        """
        These tests check the reading of field lines in various directions.
        """
        field_lines = FieldLineHandler(self.path, surface=40, lines=self.lines, 
                                       tor_range=self.tor_range, direction='backward')
        self.assertEqual(field_lines.get_field_lines().shape, (3, 4, 10))
        field_lines = FieldLineHandler(self.path, surface=40, lines=self.lines, 
                                       tor_range=self.tor_range, direction='both')
        self.assertEqual(field_lines.get_field_lines().shape, (3, 4, 10))
        field_lines_2 = FieldLineHandler(self.path, surface=40, lines=self.lines, 
                                       tor_range='-1:-500:-50', direction='backward')
        self.assertTrue(np.array_equal(field_lines.get_field_lines(), 
                                        field_lines_2.get_field_lines()))

    def test_read_multiple_files(self):
        """
        Checks reading from multiple files. Also tests correct behaviour if 
        some files are not found.
        """
        field_lines = FieldLineHandler(self.path, lines=self.lines, 
                                       tor_range=self.tor_range)
        self.assertEqual(field_lines.get_field_lines().shape, (3, 4, 10, 2))
        field_lines = FieldLineHandler(self.path, surface=(25, 30, 35),
                                       lines=self.lines, tor_range=self.tor_range)
        self.assertEqual(field_lines.get_field_lines().shape, (3, 4, 10))
        field_lines = FieldLineHandler(self.path, surface=[30, 35, 40],
                                       lines=self.lines, tor_range=self.tor_range)
        self.assertEqual(field_lines.get_field_lines().shape, (3, 4, 10, 2))

        
        