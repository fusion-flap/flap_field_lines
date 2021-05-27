# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2021

@author: lordofbejgli

This is a FLAP module that projects from device coordinates
to image coordinates for various fast cameras used in fusion devices. 
(including EDICAM and Photron HDF5)
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage
from scipy.io import readsav

class ImageProjector:
    """
    This class calculates the parameters of the projection to a camera image. 
    These consist of a projection matrix (2x3) and a translation vector (2).
    The projected points are expected to be the columns of the input for the projection.
    """
    def __init__(self, 
                 R0=1, 
                 theta0=0, 
                 z0=0, 
                 Rp=2, 
                 thetap=0, 
                 zp=0, 
                 enh=1, 
                 alpha=0, 
                 xoff=0, 
                 yoff=0,  
                 viewpoint=None, 
                 shot=None, 
                 cam=None, 
                 imsize=[1280, 1024], 
                 old = True,):
        """
        Default constructor.
        R0, theta0, z0: The coordinates of the viewpoint in cylindrical
                        device coordinates.
        Rp, thetap, zp: The coordinates of the reference point the camera
                        directly looks at.
        enh, alpha, xoff, yoff: enlargement, rotation and translation 
                                parameters on the image plane.
        viewpoint: Name of the viewpoint.
        shot: Name of the shot that was used as reference for the calibration.
        cam: Camera type.
        imsize: Dimensions of the camera image. 1st is vertical, 
                2nd is horizontal. (0, 0) is expected to be 
                the upper left corner.
        old: Legacy mode. If true, parameters are calculated by legacy method 
             based on the IDL module written by @Csega. Parameters should be 
             supplied according to his calibration.
        """
        #Transforming cylindrical coordinates to xyz
        self.__x0 = rzt_2_xyz(R0, theta0, z0)
        self.__xp = rzt_2_xyz(Rp, thetap, zp)

        #Calculating the normal vector and constant of the image plane
        self.__norm = np.squeeze(dir_vector(self.__x0, self.__xp))
        self.__d = -self.__norm @ self.__xp

        self.viewpoint = viewpoint
        self.shot = shot
        self.cam = cam
        self.__imsize = imsize

        #Wether the projected points orientation should be mirrored to match 
        #image orientation. (Due to (0,0) being in the upper left corner). 
        #Not used in "Legacy mode".
        self.__mirror = 1
        if old:
            self.__set_up_projection_old(enh, alpha, xoff, yoff)
        else:
            self.__set_up_projection(enh, alpha, xoff, yoff)
    
    @classmethod
    def from_file(cls, view, shot, cam, file):
        """
        Alternate constructor that reads precalibrated parameters from 
        a text file of specific format.
        """
        with open(file, 'r') as f:
            line = f.readline()
            is_view_valid = False
            while line[:-1] != '!!!':
                if line[:-1] == view:
                    is_view_valid = True
                line = f.readline()
            
            if not is_view_valid:
                raise ValueError('No such view.\n')

            while line[:-1] != view:
                line = f.readline()

            while line[:-1] != '!!!':
                if line[:-1] == 'shot: ' + shot:
                    line = f.readline()
                    if line[:-1] == 'cam: ' + cam:
                        line = f.readline()
                        line = line[8:-2].split(',')
                        R0 = float(line[0].split(':')[1])
                        theta0 = float(line[2].split(':')[1])
                        z0 = float(line[1].split(':')[1])
                        Rp = float(line[3].split(':')[1])
                        thetap = float(line[5].split(':')[1])
                        zp = float(line[4].split(':')[1])
                        enh = float(line[6].split(':')[1])
                        alpha = float(line[7].split(':')[1])
                        xoff = float(line[8].split(':')[1])
                        yoff = float(line[9].split(':')[1])
                        imsize = [int(line[13].split('[')[1]), int(line[14].split(']')[0])]
                        return cls(R0, theta0, z0, Rp, thetap, zp, enh, alpha, xoff, yoff, view, shot, cam, imsize)
                line = f.readline()

        raise ValueError('Parameters were not found.\n')     

    def __set_up_projection(self, enh, alpha, xoff, yoff, mirror=None):
        """
        Calculates projection matrix and translation vector,
        Rotation and projection are relative to the intersection
        of the line of view and the image plane.
        """
        base_0 = np.array([self.__norm[1], -self.__norm[0], 0])
        base_0 = base_0 / np.linalg.norm(base_0)
        base_1 = np.cross(self.__norm, base_0)

        if mirror in {-1, 1}:
            self.__mirror = mirror

        self.__projector_matrix = enh * np.array([[np.cos(alpha), self.__mirror * np.sin(alpha)], 
                                                 [-np.sin(alpha), self.__mirror * np.cos(alpha)]]) @ np.array([base_0, base_1])

        self.__offset = self.__projector_matrix @ -self.__xp + np.array([[xoff], [yoff]])

    def __set_up_projection_old(self, enh, alpha, xoff, yoff):
        """
        Legacy mode calculation of projection matrix and translation vector.
        This method is a reimplementation of the IDL routines of @Csega.
        """
        a = self.__norm[0]
        b = self.__norm[1]
        c = self.__norm[2]
        cc = np.cos(alpha)
        ss = np.sin(alpha)
        tt = 1 - np.cos(alpha)
        R = np.array([[tt*a**2 + cc, tt*a*b - ss*c, tt*a*c + ss*b],
                      [tt*a*b + ss*c, tt*b**2 + cc, tt*b*c - ss*a],
                      [tt*a*c - ss*b, tt*b*c + ss*a, tt*c**2 + cc]])

        base_0 = np.array([c, (b * c) / (a - 1), 1 + c**2 / (a - 1)])
        base_1 = np.cross(self.__norm, base_0)

        self.__projector_matrix = 190 * enh * np.array([base_1, base_0]) @ R

        self.__offset = 190 * np.array([[yoff + 2.59155], [xoff + 5.18956]])
    
    def view_parameters(self):
        """
        Return a copy of the projection parameters.
        """
        return np.copy(self.__projector_matrix), self.__offset.reshape(2,1)

    def calculate_parameters(self, x1, y1, x2, y2, p1, p2, mirror=True):
        """
        This method calculates the enlargement, rotation and offset 
        parameters of the projection by two reference points of 
        known image coordinates. Not used in "Legacy mode".
        x1, y1, x2, y2: Image coordinates of the reference points.
        p1, p2: XYZ coordinates of the reference points in column vectors.
        """
        if mirror:
            self.__set_up_projection(1, 0, 0, 0, -1)
        else:
            self.__set_up_projection(1, 0, 0, 0, 1)
        p1_i = self.calc_pixel_coord(p1)
        p2_i = self.calc_pixel_coord(p2)
        b = p2_i - p1_i
        a = np.array([x2 - x1, y2 - y1])
        L_1 = np.linalg.norm(b)
        L_2 = np.linalg.norm(a)
        alpha = np.arccos(np.float64(a @ b) / (L_2 * L_1))
        if np.cross(a, b.T) < 0:
            alpha *= -1
        enh = L_2 / L_1
        a = np.linalg.norm(p1_i) * enh
        b = np.linalg.norm(p2_i) * enh
        F = a*a - x1*x1 - y1*y1
        n = (y1 - y2) / (x2 - x1)
        m = (a*a - b*b - x1*x1 + x2*x2 - y1*y1 + y2*y2) / (2 * (x2 - x1))
        A = n*n + 1
        B = 2*n*m - 2*x1*n - 2*y1
        C = m*m - 2*x1*m - F
        O2 = (-B + np.sqrt(B*B - 4*A*C)) / (2*A)
        O1 = m + n*O2
        self.__set_up_projection(enh, alpha, O1, O2)
        p1_i = self.calc_pixel_coord(p1)
        if not np.isclose(np.array([[x1], [y1]]), p1_i, atol=0.5).all():
            O2 = (-B - np.sqrt(B*B - 4*A*C)) / (2*A)
            O1 = m + n*O2
            self.__set_up_projection(enh, alpha, O1, O2)
    
    def update_projection(self, enh=1, alpha=0, xoff=0, yoff=0):
        """
        Modifie the projection matrix and translation vector to 
        enlarge, rotate and translate the rojected points relative 
        to the center of the image.
        """
        self.__projector_matrix *= enh
        origo = np.array([[self.__imsize[1] - 1], [self.__imsize[0] - 1]]) / 2
        
        if self.__offset.ndim != 2:
            self.__offset = self.__offset.reshape(2,1)
        self.__offset = origo * (1 - enh) + enh * self.__offset

        R = np.array([[np.cos(alpha), np.sin(alpha)], 
                     [-np.sin(alpha), np.cos(alpha)]])    
        self.__projector_matrix = R @ self.__projector_matrix
        self.__offset = R @ (self.__offset - origo) + origo

        self.__offset += np.array([[xoff], [yoff]])

    def project_points(self, points):
        """
        Projects point to image plane. Input is a 3xn matrix 
        where the columns are the projected points.
        """
        points = points - self.__x0
        t = (-self.__d - np.tensordot(self.__norm, self.__x0, axes=(0,0))) / \
            np.tensordot(self.__norm, points, axes=(0,0))
        return points * t + self.__x0

    def calc_pixel_coord(self, points):
        """
        Calculates pixel coordinates of input points. Input is a 3d 
        column vector or a 3xn matrix where the columns are the 
        projected points.
        """
        if points.ndim == 3:
            self.__x0 = self.__x0.reshape(3,1,1)
            self.__offset = self.__offset.reshape(2,1,1)
        elif points.ndim == 2:
            self.__x0 = self.__x0.reshape(3,1)
            self.__offset = self.__offset.reshape(2,1)
        else:
            raise ValueError("Inappropriate number of input dimensions.")
        points = self.project_points(points)
        return np.tensordot(self.__projector_matrix, points, axes=(1,0)) + self.__offset

def dir_vector(x1, x2):
    return (x1 - x2) / np.linalg.norm(x1 - x2)

def rzt_2_xyz(r, theta, z):
    """
    Transforms cylindrical to Descartes coordinates.
    """
    return np.array([[r * np.cos(theta)], [r * np.sin(theta)], [z]])

#The rest are helper function used for testing, developement and validation. 
#They are not directly involved in the class' function.
def plot_calib_img(img, y0=115, y1=1051, x0=492, x1=1661, rotate=None, mirror=None):
    """
    Plots calib images with highlighted reference points.
    img: Path to image.
    y0, y1, x0, x1: Part of the picture to be plotted.
        Aeq20 and aeq50 needs rotate=1, aeq31 and aeq41 rotate=3.
        For aeq31 use y0=120, y1=1086, x0=627, x1=1834 insted of default.
    rotate: Number of 90° rotations.
    mirror: 1: mirror along vertical, 2: along horizontal axis.
    """
    img = mpimg.imread(img)
    img = img[y0:y1, x0:x1, :]
    plt.figure()
    extent = [0, 1279, 1023, 0]
    if rotate in {1, 2, 3}:
        img = ndimage.rotate(img, rotate*90)
        if rotate != 2:
            extent = [0, 1023, 1279, 0]
    if mirror == 2:
        img = img[:, ::-1, :]
    elif mirror == 1:
        img = img[::-1, :, :]
    plt.imshow(img, extent=extent)

def get_reference_points(file):
    """
    Read reference points for calibration from file.
    """
    points = np.loadtxt(file)
    return np.transpose(points)

def save_reference_points(file, points):
    """
    Save reference points to file.
    """
    np.savetxt(file, points.T, fmt='%.10f', delimiter='\t')

def get_field_lines(surf, folder):
    """
    Read field lines from file.
    surf: Which flux surface the plotted fiel line belong to.
    folder: Path to source files.
    """
    surf = readsav(folder + '/field_lines_tor_ang_1.85_1turn_EIM+252_w_o_limiters_w_o_torsion_w_characteristics_surf_0%d.sav' % surf)
    line_f = np.array([surf['surface'][0][4], surf['surface'][0][5], surf['surface'][0][6]])
    return line_f

def plot_field_lines(surf, folder, color='k'):
    """
    Plot 3d field lines.
    surf: Which flux surface the plotted fiel line belong to.
    folder: Path to source files.
    color: Color of the plotted lines.
    """
    line_f = get_field_lines(surf, folder)

    plt.figure()
    plt.gca(projection='3d')

    for i in range(0, 360, 30):
        plt.plot(line_f[0, i, :], line_f[1, i, :], line_f[2, i, :], color=color)
    
def plot_view(R0, theta0, z0, Rp, thetap, zp, color='r'):
    """
    Plot arrows along camera views.
    """
    x0 = rzt_2_xyz(R0, theta0, z0)
    xp = rzt_2_xyz(Rp, thetap, zp)
    plt.quiver(x0[0],x0[1],x0[2], xp[0] - x0[0], xp[1] - x0[1], xp[2] - x0[2], arrow_length_ratio=0.05, color=color)

def get_flux_surface(number, r, folder):
    """
    Return radial cross-section of selected flux surface.
    """
    surf = readsav(folder + '/field_lines_tor_ang_1.85_1turn_EIM+252_w_o_limiters_w_o_torsion_w_characteristics_surf_%s.sav' % number)
    line_f = np.array([surf['surface'][0][4][:,r], surf['surface'][0][5][:,r], surf['surface'][0][6][:,r]])
    return line_f

def plot_surface_2d(view, number, r, folder, color):
    """
    Plot projected cross-section of flux surface.
    """
    surf = get_flux_surface(number, r, folder)
    surf = view.calc_pixel_coord(surf)
    plt.plot(surf[0,:], surf[1,:], color)

def plot_surface_3d(number, r, folder, color):
    """
    Plot radial cross-section of a selected flux surface in 3d.
    """
    surf = get_flux_surface(number, r, folder)
    plt.plot(surf[0,:], surf[1,:], surf[2,:], color)

def plot_flux_surfaces(view, r, folder, color='k'):
    """
    Plot projected radial cross-sections of all flux surfaces.
    """
    for i in range (1,10):
        file_number = '00%d' % i
        plot_surface_2d(view, file_number, r, folder, color)
    for i in range (10,95):
        file_number = '0%d' % i
        plot_surface_2d(view, file_number, r, folder, color)

def plot_flux_surfaces_3d(r, folder, color='k'):
    """
    Plot radial cross-section of all flux surfaces in 3d.
    """
    for i in range (1,10):
        file_number = '00%d' % i
        plot_surface_3d(file_number, r, folder, color)
    for i in range (10,95):
        file_number = '0%d' % i
        plot_surface_3d(file_number, r, folder, color)
