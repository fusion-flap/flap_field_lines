# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2021

@author: lordofbejgli
"""

import unittest
import numpy as np

from flap_field_lines.image_projector import *

class TestOld(unittest.TestCase):
    """
    This class of tests checks wether the legacy mode projects the reference 
    points of each camera view to the correct values.
    """
    def setUp(self):
        self.file='views2.txt'

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

class TestNew(unittest.TestCase):
    """
    This class of tests checks wether the acquiring of the projection 
    parameters and the projection itself is done correctly on for each
    camera view.
    """
    def compare_points(self, R0, theta0, z0, Rp, thetap, zp, view):
        """
        Helper function. Compares the projected coordinates 
        of reference points with the expected values. 
        """
        points = get_reference_points('tests/integration/fixtures/%s.dat' % view)
        points_ref = get_reference_points('tests/integration/fixtures/%s_ref.dat' % view)
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, old=False)

        view.calculate_parameters(points_ref[0, 10], points_ref[1, 10],
                                  points_ref[0, 14], points_ref[1, 14], 
                                  points[:, 10:11], points[:, 14:15])

        points = view.calc_pixel_coord(points)

        return np.allclose(points, points_ref)

    def test_aeq20(self):
        view = 'aeq20'
        R0 = 6.34811
        z0 = -0.649178
        theta0 = 1.21099
        Rp = 6.21000
        zp = 0.810000
        thetap = -0.0400000
        self.assertTrue(self.compare_points(R0, theta0, z0, Rp, thetap, zp, view))
    
    def test_aeq31(self):
        view = 'aeq31'
        R0 = 6.44669
        z0 = 0.665888
        theta0 = 2.54492
        Rp = 7.44
        zp = 0.48
        thetap = 4.01
        self.assertTrue(self.compare_points(R0, theta0, z0, Rp, thetap, zp, view))
        
    def test_aeq41(self):
        view = 'aeq41'
        R0 = 6.34720
        z0 = 0.643984
        theta0 = 3.81564
        Rp = 6.31000
        zp = -0.560000
        thetap = -1.15000
        self.assertTrue(self.compare_points(R0, theta0, z0, Rp, thetap, zp, view))

    def test_aeq50(self):
        view = 'aeq50'
        R0 = 6.34459
        z0 = -0.643974
        theta0 = -1.30240
        Rp = 6.31000
        zp = 0.560000
        thetap = 3.72000
        self.assertTrue(self.compare_points(R0, theta0, z0, Rp, thetap, zp, view))

class Test3dInput(unittest.TestCase):
    """
    This class checks if proper results are returned in case of the input 
    array being 3d.
    """
    def test_calc_pixel_coord_3d(self):
        path = 'tests/integration/fixtures/'
        
        points = np.load(path + '3d_test_array.npy', 
                         allow_pickle=False)
        points_ref = np.load(path + '3d_test_array_ref.npy', 
                             allow_pickle=False)

        view = ImageProjector.from_file('W7X-AEQ31', '20160218', 'edicam', 
                                        'views2.txt')
        
        points = view.calc_pixel_coord(points)

        self.assertTrue(np.allclose(points, points_ref))

if __name__ == '__main__':
    unittest.main(verbosity=2)
    