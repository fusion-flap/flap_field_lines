# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2021

@author: lordofbejgli
"""

import unittest
from matplotlib.pyplot import get
import numpy as np

from flap_field_lines.image_projector import *

class TestOld(unittest.TestCase):
    
    def setUp(self):
        self.file='tests/integration/fixtures/views2.txt'

    def compare_points(self, view, shot, cam):
        """
        Helper function. Compares the projected coordinates 
        of reference points with the expected values. 
        """
        points = get_reference_points('tests/integration/fixtures/%s.dat' % view)
        points_ref = get_reference_points('tests/integration/fixtures/%s_ref.dat' % view)
        view = ImageProjector.from_file('W7X-%s' % view.upper(), shot, cam, self.file)

        points = view.calc_pixel_coord(points)
        
        return np.allclose(points, points_ref)

    def test_aeq20(self):
        view = 'aeq20'
        shot = '20160308'
        cam = 'edicam'
        self.assertTrue(self.compare_points(view, shot, cam))

    def test_aeq31(self):
        view = 'aeq31'
        shot = '20160218'
        cam = 'edicam'
        self.assertTrue(self.compare_points(view, shot, cam))

    def test_aeq50(self):
        view = 'aeq50'
        shot = '20160308'
        cam = 'edicam'
        self.assertTrue(self.compare_points(view, shot, cam))

    def test_aeq41(self):
        view = 'aeq41'
        shot = '20160308'
        cam = 'edicam'
        points = get_reference_points('tests/integration/fixtures/%s.dat' % view)
        points_ref = get_reference_points('tests/integration/fixtures/%s_ref.dat' % view)
        view = ImageProjector.from_file('W7X-%s' % view.upper(), shot, cam, self.file)
        
        #This view needs and extra 180 degree rotation to match the calib image.
        view.update_projection(alpha=np.pi)

        points = view.calc_pixel_coord(points)

        self.assertTrue(np.allclose(points, points_ref))



if __name__ == '__main__':
    unittest.main(verbosity=2)