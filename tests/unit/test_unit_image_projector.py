# -*- coding: utf-8 -*-
"""
Created on Tue May 25 2021

@author: lordofbejgli
"""

from os import XATTR_REPLACE
import unittest
import numpy as np
from numpy.lib.scimath import sqrt

from flap_field_lines.image_projector import *

class TestAccessories(unittest.TestCase):
    """
    This class tests two helper functions of the ImageProjector class.
    """
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

class TestSetUpProjection(unittest.TestCase):
    """
    These unit tests test the set_up_projection_old function by comparing its
    results to known correct values for 4 viewports.
    """
    def test_set_up_projection_old_aeq20(self):
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
        ref_M = np.array([[-255.80189788, -171.61517908, -32.05790618], [59.8712411, -32.81234059, -302.0809077]])
        ref_O = np.array([[2059.8945], [509.1164]])
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, l0, gamma, xoffset, yoffset)
        matrix, offset = view.view_parameters()
        self.assertTrue(np.allclose(ref_M, matrix), msg='Projection matrix is wrong.')
        self.assertTrue(np.allclose(ref_O, offset), msg='Offset is wrong.')
        
    
    def test_set_up_projection_old_aeq31(self):
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
        ref_M = np.array([[-248.92937168, -13.53482756, -27.41562252], [27.09639828, 6.5141635, -249.24684727]])
        ref_O = np.array([[-748.3055], [528.1164]])
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, l0, gamma, xoffset, yoffset)
        matrix, offset = view.view_parameters()
        self.assertTrue(np.allclose(ref_M, matrix), msg='Projection matrix is wrong.')
        self.assertTrue(np.allclose(ref_O, offset), msg='Offset is wrong.')

        
    def test_set_up_projection_old_aeq41(self):
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
        ref_M = np.array([[74.68264825, 283.39567647, 44.29844928], [33.3574145, -54.04568694, 289.51564141]])
        ref_O = np.array([[2012.3945], [425.5164]])
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, l0, gamma, xoffset, yoffset)
        matrix, offset = view.view_parameters()
        self.assertTrue(np.allclose(ref_M, matrix), msg='Projection matrix is wrong.')
        self.assertTrue(np.allclose(ref_O, offset), msg='Offset is wrong.')

    def test_set_up_projection_old_aeq50(self):
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
        ref_M = np.array([[106.53871141, 289.68105731, -25.46523136], [-55.14896823, -6.50117809, -304.68084283]])
        ref_O = np.array([[2080.7945], [528.1164]])
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, l0, gamma, xoffset, yoffset)
        matrix, offset = view.view_parameters()
        self.assertTrue(np.allclose(ref_M, matrix), msg='Projection matrix is wrong.')
        self.assertTrue(np.allclose(ref_O, offset), msg='Offset is wrong.')

    def test_set_up_projection(self):
        """
        Compares calculated values to known correct ones for the given input.
        """
        view = ImageProjector(old=False)
        ref_M = np.array([[0, 1, 0], [0, 0, -1]])
        ref_O = np.array([[0], [0]])
        matrix, offset = view.view_parameters()
        self.assertTrue(np.allclose(ref_M, matrix), msg='Default projection matrix is wrong.')
        self.assertTrue(np.allclose(ref_O, offset), msg='Default offset is wrong.')
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
        ref_M = np.array([[0.55187711, 0.87145393, 1.2621013], [-0.30675626, -1.251502, 0.99827018]])
        ref_O = np.array([[2.80529554], [1.86299051]])
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, l0, gamma, xoffset, yoffset, old=False)
        matrix, offset = view.view_parameters()
        self.assertTrue(np.allclose(ref_M, matrix), msg='Projection matrix is wrong.')
        self.assertTrue(np.allclose(ref_O, offset), msg='Offset is wrong.')


class TestImageProjector(unittest.TestCase):
    """
    This class test the various member functions of the ImageProjector class.
    """

    def setUp(self):
        """
        Sets up a custom view for later tests.
        """
        self.R0 = 6.31
        self.z0 = 0.57
        self.theta0 = -np.pi / 3
        self.Rp = 6.274
        self.zp = 0.023
        self.thetap = 5 * np.pi / 12
        self.x0 = rzt_2_xyz(self.R0, self.theta0, self.z0)
        self.xp = rzt_2_xyz(self.Rp, self.thetap, self.zp)
        self.view = ImageProjector(self.R0, self.theta0, self.z0, 
                                   self.Rp, self.thetap, self.zp, old=False)

    def test_project_points_1d_1(self):
        """
        Checks wether the projection of a single point works well.
        """
        x = np.array([[2.74], [4.13], [0.45]])

        x_pro = self.view.project_points(x)

        self.assertEqual(x.shape, x_pro.shape, 
                         msg="Input and output shapes do not match.")
        self.assertTrue(np.allclose(self.xp, self.view.project_points(self.xp)), 
                        msg="Intersection of projection line and plane is not \
                             projected to itself.")
        self.assertTrue(np.allclose(dir_vector(x, self.x0), dir_vector(x_pro, x)),
                        msg="Projected point is not in line with the original \
                             and the viewport.")
        self.assertAlmostEqual(float(dir_vector(self.xp, self.x0).T @ dir_vector(x_pro, self.xp)), 0, 
                               msg="Projected point is not on the plane.")

    def test_project_points_1d_2(self):
        """
        Checks wether the projection of a single point that is behind 
        the plane works well.
        """
        x = np.array([[3.321], [8.5], [0.234]])

        x_pro = self.view.project_points(x)

        self.assertEqual(x.shape, x_pro.shape, 
                         msg="Input and output shapes do not match.")
        self.assertTrue(np.allclose(self.xp, self.view.project_points(self.xp)), 
                        msg="Intersection of projection line and plane is not \
                             projected to itself.")
        self.assertTrue(np.allclose(dir_vector(x, self.x0), -dir_vector(x_pro, x)),
                        msg="Projected point is not in line with the original \
                             and the viewport.")
        self.assertAlmostEqual(float(dir_vector(self.xp, self.x0).T @ dir_vector(x_pro, self.xp)), 0, 
                               msg="Projected point is not on the plane.")

    def test_project_points_2d(self):
        """
        Checks wether the projection of multiple points in a 2d array works well.
        """
        x = np.array([[2.74, 3.321, 0.98], [4.13, 5.7, 1.43], [0.234, 0.45, 0]])

        x2 = self.view.project_points(x)

        self.assertEqual(x.shape, x2.shape, 
                         msg="Input and output shapes do not match.")
        self.assertTrue(np.allclose((x - self.x0) / np.linalg.norm(x - self.x0, axis=0), 
                                    (x2 - x) / np.linalg.norm(x2 - x, axis=0)),
                                    msg="Projected points are not in line with \
                                        the original and the viewport.")
        self.assertTrue(np.allclose((self.xp - self.x0).T @ (x2 - self.xp), np.array([0, 0, 0])), 
                        msg="Projected points are not on the plane.")

    def test_calc_pixel_coord_1(self):
        """
        Checks if the result is the proper shape.
        Checks if the intersection of the projection line and projection plane 
        is returned as (0, 0) if no offset is given.
        Checks if the projection keeps length and orientation.
        """
        x = np.array([[2.74, 3.321, 0.98], [4.13, 5.7, 1.43], [0.234, 0.45, 0]])

        x_pro = self.view.project_points(x)

        x = self.view.calc_pixel_coord(x)
        self.assertEqual(x.shape, (2, 3), msg="Incorrect output shape.")

        self.assertTrue(np.allclose(self.view.calc_pixel_coord(self.xp), 
                                    np.array([[0], [0]])), 
                                    msg="Interjection is not projected to (0, 0).")

        x_pro = x_pro - self.xp
        self.assertTrue(np.allclose(np.linalg.norm(x_pro, axis=0), 
                                    np.linalg.norm(x, axis=0)), 
                                    msg="Projection doesn't keep length.")

        self.assertAlmostEqual(np.arccos(x_pro[:, 0] @ x_pro[:, 1] / 
                               np.prod(np.linalg.norm(x_pro[:, 0:2], axis=0))), 
                               np.arccos(x[:, 0] @ x[:, 1] / 
                               np.prod(np.linalg.norm(x[:, 0:2], axis=0))),
                               msg="Orientation is not preserved.")
    
    def test_calc_pixel_coord_2(self):
        """
        Tests if other projection parameters work as intended.
        """
        enh = 3.6
        alpha = 2 * np.pi / 5
        offset = np.array([[4.3], [7.2]])

        x = np.array([[3.321], [8.5], [0.234]])
        x1 = self.view.calc_pixel_coord(x)

        view2 = ImageProjector(self.R0, self.theta0, self.z0, 
                               self.Rp, self.thetap, self.zp, 
                               enh, alpha, offset[0, 0], offset[1, 0], 
                               old=False)

        x2 = view2.calc_pixel_coord(x)

        self.assertTrue(np.allclose(view2.calc_pixel_coord(self.xp), offset), 
                                    msg="Interjection is not projected to offset.")
        self.assertAlmostEqual(np.linalg.norm(x2 - offset) / np.linalg.norm(x1), 
                               enh, msg="Enlargement doesn't work properly.")
        self.assertAlmostEqual(np.arccos(x1.T @ (x2 - offset) /  
                              (np.linalg.norm(x2 - offset) * np.linalg.norm(x1))),
                              alpha, msg="Rotation doesn't work properly.")

class TestProjectionUpdate(unittest.TestCase):
    """
    This class tests te update_projection function. It uses the 
    reference points for AEQ31.
    """
    def setUp(self):
        points = get_reference_points('tests/integration/fixtures/aeq31.dat')
        points_ref = get_reference_points('tests/integration/fixtures/aeq31_ref.dat')
        
        self.points = points[:, 0:2]
        self.points_ref = points_ref[:, 0:2]
        self.center = np.array([[511.5], [639.5]])
        self.enh = 1.5
        self.alpha = np.pi/3
        self.offset = np.array([[100], [200]])

    def test_update_projection_old(self):
        """
        This test checks if enhancement, rotation and translation works well.
        """        
        view = ImageProjector.from_file('W7X-AEQ31', '20160218', 'edicam', 
                                        'views2.txt')
        
        view.update_projection(self.enh, self.alpha, 
                               self.offset[0, 0], self.offset[1, 0])
        points_2 = view.calc_pixel_coord(self.points)

        self.assertAlmostEqual(np.linalg.norm(points_2[:, 0] - points_2[:, 1]) / 
                               np.linalg.norm(self.points_ref[:, 0] - 
                                              self.points_ref[:, 1]), 
                               self.enh, msg="Enlargement doesn't work properly.")
        v1 = points_2[:, 0:1] - self.offset - self.center
        v2 = self.points_ref[:, 0:1] - self.center
        self.assertAlmostEqual(float(np.arccos(v1.T @ v2 / 
                        (np.linalg.norm(v1) * np.linalg.norm(v2)))), 
                        self.alpha, msg="Rotation doesn't work properly.")

    def test_update_projection(self):
        """
        This test checks if updating enhancement, rotation and translation works well, 
        by comparing the modified projections results to the original.
        """        
        R0 = 6.44669
        z0 = 0.665888
        theta0 = 2.54492
        Rp = 7.44
        zp = 0.48
        thetap = 4.01
        view = ImageProjector(R0, theta0, z0, Rp, thetap, zp, old=False)

        view.calculate_parameters(self.points_ref[0, 0], self.points_ref[1, 0],
                                  self.points_ref[0, 1], self.points_ref[1, 1], 
                                  self.points[:, 0:1], self.points[:, 1:2])
        
        view.update_projection(self.enh, self.alpha, 
                               self.offset[0, 0], self.offset[1, 0])
        points_2 = view.calc_pixel_coord(self.points)

        self.assertAlmostEqual(np.linalg.norm(points_2[:, 0] - points_2[:, 1]) / 
                               np.linalg.norm(self.points_ref[:, 0] - 
                                              self.points_ref[:, 1]), 
                               self.enh, msg="Enlargement doesn't work properly.")
        v1 = points_2[:, 0:1] - self.offset - self.center
        v2 = self.points_ref[:, 0:1] - self.center
        self.assertAlmostEqual(float(np.arccos(v1.T @ v2 / 
                        (np.linalg.norm(v1) * np.linalg.norm(v2)))), 
                        self.alpha, msg="Rotation doesn't work properly.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
