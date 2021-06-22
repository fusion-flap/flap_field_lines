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
        #Write flux surface files location here
        self.path = '/media/data/w7x_flux_surfaces'
        self.config = 'EIM'

    