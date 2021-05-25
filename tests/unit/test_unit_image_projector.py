# -*- coding: utf-8 -*-
"""
Created on Tue May 25 2021

@author: lordofbejgli
"""

import unittest
import numpy as np
from numpy.lib.scimath import sqrt

from flap_field_lines.image_projector import *

class TestAccessories(unittest.TestCase):

    def test_rzt_2_xyz(self):
        r = 2.513
        theta = 1.2345
        z = 2.31
        x = rzt_2_xyz(r, theta, z)
        self.assertEqual(x.shape, (3, 1), msg="Output shape is wrong.")
        self.assertEqual(x[2], z, msg="Z coordinate is wrong.")
        self.assertEqual(sqrt(x[0]**2 + x[1]**2), r, msg="Output xy projection length is wrong.")
        self.assertEqual(np.linalg.norm(x), sqrt(r**2 + z**2), msg="Output length is wrong.")
        self.assertEqual(np.arctan(x[1] / x[0]), theta, msg="Output angle is wrong.")

    def test_dir_vector(self):
        x0 = np.array([[1.5], [-2.1], [3.4]])
        x1 = np.array([[0.7], [-4.6], [3.2]])
        diff_v = np.array([[0.8], [2.5], [0.2]])
        v = dir_vector(x0, x1)
        self.assertEqual(v.shape, (3, 1), msg="Output shape is wrong.")
        self.assertAlmostEqual(np.linalg.norm(v), 1, msg="Vector is not properly normed.")
        self.assertAlmostEqual(np.linalg.norm(diff_v), float(v.T @ diff_v), msg="Vector is not properly directed.")

class TestImageProjectorOld(unittest.TestCase):

    def test_old_aeq20(self):
        R0 = 6.34811
        z0 = -0.649178
        theta0 = 1.21099
        Rp = 6.21000
        zp = 0.810000
        thetap = -0.0400000
        l0 = 1.63000
        gamma = 3.14000
        xoffset = -2.51000
        yoffset = 8.25000
        ref_M = np.array([[-255.80189788, -171.61517908,  -32.05790618], [59.8712411 ,  -32.81234059, -302.0809077]])
        ref_O = np.array([[2059.8945], [509.1164]])
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, l0, gamma, xoffset, yoffset)
        matrix, offset = view.view_parameters()
        self.assertTrue(np.allclose(ref_M, matrix))
        self.assertTrue(np.allclose(ref_O, offset))
        
    
    def test_old_aeq31(self):
        R0 = 6.44669
        z0 = 0.665888
        theta0 = 2.54492
        Rp = 7.44
        zp = 0.48
        thetap = 4.01
        l0 = 1.32000
        gamma = 3.27000
        xoffset = -2.41000
        yoffset = -6.53000
        ref_M = np.array([[-248.92937168,  -13.53482756,  -27.41562252], [27.09639828,    6.5141635 , -249.24684727]])
        ref_O = np.array([[-748.3055], [528.1164]])
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, l0, gamma, xoffset, yoffset)
        matrix, offset = view.view_parameters()
        self.assertTrue(np.allclose(ref_M, matrix))
        self.assertTrue(np.allclose(ref_O, offset))

        
    def test_old_aeq41(self):
        R0 = 6.34720
        z0 = 0.643984
        theta0 = 3.81564
        Rp = 6.31000
        zp = -0.560000
        thetap = -1.15000
        l0 = 1.56000
        gamma = 0.170000
        xoffset = -2.95000
        yoffset = 8.00000
        ref_M = np.array([[74.68264825, 283.39567647,  44.29844928], [33.3574145 , -54.04568694, 289.51564141]])
        ref_O = np.array([[2012.3945], [425.5164]])
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, l0, gamma, xoffset, yoffset)
        matrix, offset = view.view_parameters()
        self.assertTrue(np.allclose(ref_M, matrix))
        self.assertTrue(np.allclose(ref_O, offset))

    def test_old_aeq50(self):
        R0 = 6.34459
        z0 = -0.643974
        theta0 = -1.30240
        Rp = 6.31000
        zp = 0.560000
        thetap = 3.72000
        l0 = 1.63000
        gamma = -2.24000
        xoffset = -2.41000
        yoffset = 8.36000
        ref_M = np.array([[106.53871141,  289.68105731,  -25.46523136], [-55.14896823,   -6.50117809, -304.68084283]])
        ref_O = np.array([[2080.7945], [528.1164]])
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, l0, gamma, xoffset, yoffset)
        matrix, offset = view.view_parameters()
        self.assertTrue(np.allclose(ref_M, matrix))
        self.assertTrue(np.allclose(ref_O, offset))

class TestImageProjector(unittest.TestCase):

    def test_1(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main(verbosity=2)