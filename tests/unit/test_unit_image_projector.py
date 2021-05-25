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
