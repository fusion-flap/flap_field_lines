# -*- coding: utf-8 -*-
"""
Created on Tue May 31 2021

@author: lordofbejgli
"""

import numpy as np
import os
import glob

from scipy.io import readsav

from .errors import *

class FieldLineHandler:
    """
    This class reads and stores field lines of various flux surfaces from the 
    given .sav files. In its current form, it's equipped to handle W7X field 
    lines. One .sav file contains the forward and backward calculated field 
    lines of one flux surface. Typically there is 360 field lines per surface, 
    which are 3651 bin long. Results are stored in an numpy array. It can be 
    accessed via the return_field_lines() function. The magnetic field and 
    its gradient may optionally be read from these file as well and store in 
    separate arrays. The constructor reads some data about the surfaces from 
    the fs_info.sav file. 

    Stored numpy array may have 2, 3 or 4 dimensions. These are:
        - 1st dimension: 3 long, x, y, z coordinates
        - last dimension: flux surface coordinate if there's more
        - 2nd and 3rd: poloidal (if multiple lines are selected per surface) 
          and toroidal coordinates
    These arrays can be directly projected to image coordinates by the 
    ImageProjector class.
    """
    #W7X magnetic configs
    __valid_configs = ('EIM', 'FTM', 'KJM001', 'EJM','DBM')

    def __init__(self, 
                 path=None, 
                 configuration=None):
        """
        Constructor. Inputs:
        path: path to fs_info.sav. If None is given, default vaue is used
        configuration: W7X magnetic config name. Currently there are 3 valid 
                       options.
        
        Raises: NoFsInfoError and WrongConfigurationError.
        """
        if path is None:
            #default path
            self.path = '/data/W7-X/processed_data/flux_surfaces/%s+252_detailed_w_o_limiters_w_o_torsion'
            if configuration in self.__valid_configs:
                path = self.path = self.path % configuration 
                #default path for fs_info.sav. Raise error if not found
                if not os.path.isfile(path + '/fs_info.sav'):
                    raise NoFsInfoError
            else:
                raise WrongConfigurationError(configuration)
        else:
            #if path for fs_info was give, set path for its root folder
            self.path = path
            #if self.path/field_lines exists, it is used as default
        if os.path.exists(os.path.join(self.path, 'field_lines')):
            self.path = os.path.join(self.path, 'field_lines')
        if os.path.isfile(os.path.join(path, 'fs_info.sav')):
            self.__fs_info = self.__read_fs_info(os.path.join(path, 'fs_info.sav'))
        else:
            self.__fs_info = None
        if len(glob.glob(self.path + '/*field_line*')) == 0:
            raise FileNotFoundError
        self.configuration = configuration
        self.__field_lines = None
        self.__B = None
        self.__gradB = None
        self.surfaces = []
        self.surface_files = []
        self.read_files = []
        self.lines = None
        self.tor_range = None
        self.direction = None

    def __read_fs_info(self, file):
        '''
        Reads data from fs_info file. It takes the iota of each eligible flux 
        surface for the given magnetic configuration. Effective radius of the 
        surfaces. The surface number of the separatrix for the main plasma and 
        islands. Names of the islands. Flag of which surfaces belongs to which 
        island.
        '''
        fs_info = readsav(file)
        needed_info = {}
        needed_info['iota'] = fs_info['fs_info'][0][3]
        needed_info['reff'] = fs_info['fs_info'][0][4]
        needed_info['separatrix'] = fs_info['fs_info'][0][6]
        needed_info['names'] = fs_info['fs_info'][0][7]
        needed_info['names'][0] = b'main plasma'
        needed_info['flags'] = fs_info['fs_info'][0][8]
        return needed_info

    def update_read_parameters(self, 
                 path=None, 
                 surfaces=None, 
                 lines=None, 
                 tor_range=None, 
                 direction='forward',
                 drop_data=True):
        """
        Updates the parameters for reading data. Here can be set which flux 
        surfaces are to be read, which lines and at what toroidal range. Files 
        of flux surfaces can be directly given as a list of strings in 
        'surfaces', or can be infered from the specified surface numbers (also 
        in 'surfaces') and 'self.path'. If 'path' is specified here, self.path 
        is set to it, otherwise it is derived from the path of the fs_info file. 
        Surfaces files are looked for in fs_info's folder, unless there's a 
        'field_line' folder is present, in which case there.

        parameters:
        path: where to look for the surface files; ignored if specific files 
            (with full path) are given in surfaces
        surfaces: selection of surfaces to read. it can be of various format:
            - single int
            - enumeration of ints in an iterable container
            - string: 
                - ':' to select all available
                - python range format: 'from:to(:by)'
            - list of string, each is a filename with full path
        lines: which field lines to read from the surfaces (usually 360 per 
            field line). format:
            - single int
            - enumeration of ints in an iterable container
            - string: 
                - ':' to select all available
                - from:to(:by)
        tor_range: which part of field lines to read (one line is usually 3651 
            bin long). same format as lines
        direction: which direction of field lines to read. Can be 
            forward, backward or both
        drop_data: overwrite if True or extend existing data if False
        """
        if direction in ('forward', 'backward', 'both'):
            #if the direction is changed, drop existing data and reread all
            if direction != self.direction:
                self.drop_data()
                self.direction = direction
        else:
            raise WrongDirectionError

        if path:
            #if path is given, self.path is overwritten
            self.path = path

        if surfaces is not None:
            try:
                #if surfaces is given, extract a list of surfaces to be read
                surfaces = process_selection(surfaces)
            except (TypeError, ValueError):
                #if an error of these kind occurs, check if surfaces is a list 
                #of filenames. if not reraise error, otherwise use it.
                if not os.path.isfile(surfaces[0]):
                    raise
                surf_files = surfaces
                #create a list of surfaces from the filenames
                surfaces = self.create_surf_list(surfaces)
            else:
                #if the list of surfaces is successfully processed, use it to 
                #create a list of files to be read
                surf_files, surfaces = self.create_surf_file_list(surfaces)
        elif not self.surfaces:
            #if surfaces is not specified and there are no surfaces already 
            #selected, select all according to fs_info
            surf_files, surfaces = self.create_surf_file_list(range(len(self.__fs_info['iota'])))
        else:
            #if nothing is given and there are already selected surfaces, do 
            #nothing
            surf_files = []
            surfaces = []

        if drop_data:
            if surfaces:
                self.surfaces = surfaces
                self.surface_files = surf_files
            self.drop_data()
        else:
            #extend the list of files to read if drop_data is False
            self.surfaces += surfaces
            self.surface_files += surf_files
            self.read_files += [True for i in range(len(surfaces))]

        #if new lines or range is to be read, drop all data and reread it along 
        #with the new data
        if lines:
            lines = process_selection(lines)
            if lines != self.lines:
                self.lines = lines
                self.drop_data()

        if tor_range:
            tor_range = process_selection(tor_range)
            if tor_range != self.tor_range:
                self.tor_range = tor_range
                self.drop_data()

    def load_data(self, getB=False, getGradB=False):
        if self.__B is not None:
            getB = True
        elif getB:
            self.drop_data()

        if self.__gradB is not None:
            getGradB = True
        elif getGradB:
            self.drop_data()
        
        first = self.read_files.index(True)

        field_lines, B, grad_B = self.__read_surf_files(first, getB, getGradB)

        if first == 0:
            self.__field_lines = field_lines
            self.__B = B
            self.__gradB = grad_B
        else:
            if self.__field_lines.ndim < field_lines.ndim:
                self.__field_lines = self.__field_lines[..., np.newaxis]
                if self.__B is not None:
                    self.__B = self.__B[..., np.newaxis]
                if self.__gradB is not None:
                    self.__gradB = self.__gradB[..., np.newaxis]

            self.__field_lines = np.concatenate((self.__field_lines, 
                                                 field_lines), axis=-1)
            if getB:
                self.__B = np.concatenate((self.__B, B), axis=-1)
            if getGradB:
                self.__gradB = np.concatenate((self.__gradB, grad_B), axis=-1)

        self.read_files[first:] = [False for i in range(first,len(self.surfaces))]

    def drop_data(self):
        self.read_files = [True for i in range(len(self.surfaces))]
        self.__field_lines = None
        self.__B = None
        self.__gradB = None
    
    def create_surf_file_list(self, surfs):
        file = os.path.join(self.path, 'field_lines_tor_ang_1.85_1turn_%s+252_w_o_limiters_w_o_torsion_w_characteristics_surf_')
        file = file % self.configuration + '%s.sav'
        file_list = []
        surf_list = []

        if surfs is None:
            surfs = range(len(self.return_fs_info()['iota']))

        for i in surfs:
            string_no = str(i)
            if i < 10:
                string_no = '00' + string_no
            elif i < 100:
                string_no = '0' + string_no
            if os.path.isfile(file % string_no):
                file_list.append(file % string_no)
                surf_list.append(i)
        
        return file_list, surf_list

    def create_surf_list(self, file_list):
        surfs = []
        for i in file_list:
            surfs.append(int(i.split('_')[-1][0:3]))
        return surfs

    def __read_surf_files(self, index, get_B=False, get_gradB=False):
        surf = readsav(self.surface_files[index])
         #if no lines are specified, chooses all
        if self.lines is None:
            self.lines = range(len(surf['surface'][0][4]))

        #if no toroidal range is specified, chooses all
        if self.tor_range is None:
            if self.direction == 'both':
                self.tor_range = range(2*len(surf['surface'][0][4][0]))
            else:
                self.tor_range = range(len(surf['surface'][0][4][0]))

        field_lines = self.__extract_data_from_surf(surf, 4)

        if get_B:
            B = self.__extract_data_from_surf(surf, 10)
        else:
            B = None

        if get_gradB:
            gradB = self.__extract_data_from_surf(surf, 16)
        else:
            gradB = None
        
        if len(self.surface_files) > 1:
            field_lines = field_lines[..., np.newaxis]
            if get_B:
                B = B[..., np.newaxis]
            if get_gradB:
                gradB = gradB[..., np.newaxis]
        else:
            return field_lines, B, gradB

        for file in self.surface_files[index+1:]:
            surf = readsav(file)

            new = self.__extract_data_from_surf(surf, 4)
            field_lines = np.concatenate((field_lines, 
                                          new[..., np.newaxis]), 
                                          axis=-1)
            if get_B:
                new = self.__extract_data_from_surf(surf, 10)
                B = np.concatenate((B, new[..., np.newaxis]), axis=-1)
            if get_gradB:
                new = self.__extract_data_from_surf(surf, 16)
                gradB = np.concatenate((gradB, new[..., np.newaxis]), axis=-1)

        return field_lines, B, gradB

    def __extract_data_from_surf(self, surf, index_no):
        """
        Returns requested data from a flux surface file. Acts as part of the 
        constructor, not used by itself. Takes the surface number and the 
        processed selections as input.
        Raises "NoSurfaceFileError" if file not found.
        """
        data = []
        if self.direction == 'forward':
            #reads forward calculated field lines
            data = np.array([surf['surface'][0][index_no][self.lines], 
                             surf['surface'][0][index_no + 1][self.lines], 
                             surf['surface'][0][index_no + 2][self.lines]])
        elif self.direction == 'backward':
            #reads backward calculated field lines
            data = np.array([surf['surface'][0][index_no + 3][self.lines], 
                             surf['surface'][0][index_no + 4][self.lines], 
                             surf['surface'][0][index_no + 5][self.lines]])
        elif self.direction == 'both':
            #reads both. backward lines are erversed and placed in front of 
            #forward lines
            data = np.array([surf['surface'][0][index_no][self.lines], 
                             surf['surface'][0][index_no + 1][self.lines], 
                             surf['surface'][0][index_no + 2][self.lines]])
            data = np.concatenate((np.array([surf['surface'][0][index_no + 3][self.lines, -1::-1], 
                                             surf['surface'][0][index_no + 4][self.lines, -1::-1], 
                                             surf['surface'][0][index_no + 5][self.lines, -1::-1]]), 
                                             data), axis=2)
        return data[:, :, self.tor_range]

    def return_field_lines(self):
        """
        Returnes stored data
        """
        return self.__field_lines
                
    def return_B(self):
        """
        Returnes stored data
        """
        return self.__B

    def return_gradB(self):
        """
        Returnes stored data
        """
        return self.__gradB

    def return_fs_info(self):
        """
        Returnes stored data
        """
        return self.__fs_info

    def return_surface_files(self):
        return self.surface_files

    def return_surfaces(self):
        return self.surfaces

def iter_2_array(selected):
    #if selection is iterable, put all elements into a list
    a = []
    for x in selected:
        if isinstance(x, int):
            a.append(x)
        else:
            raise TypeError('Field line selection should only have integers.')
    return a

def str_2_range(selected):
    selected = selected.split(':')
    try:
        first = int(selected[0])
        last = int(selected[1])
    except (ValueError, IndexError):
        raise ValueError('Input string should be of format from:to(:by).')
    if len(selected) == 3:
        #step bw elements is optional
        step = int(selected[2])
        return range(first, last, step)
    elif len(selected) > 3:
        raise ValueError('Input string should be of format from:to(:by).')
    else:
        return range(first, last)

def process_selection(selected):
    """
    Processes selection. Either returnes a list in case of single or mutiple 
    specific selection, or a range if a range is specified. Returns None, in 
    the case of a single ':' as input.
    Raises ValueError for wrong string format and TypeError if non-ints are given, 
    or something unexpected.
    """
    if isinstance(selected, int):
        #returns 1 long list for singe selection
        return [selected]
    elif isinstance(selected, str):
        #returns range if a string is specified. string is broken up by ":"
        if selected == ':':
            return None
        selected = selected.split(',')
        a = []
        for sel in selected:
            try:
                a += [int(sel)]
            except ValueError:
                a += iter_2_array(str_2_range(sel))
        return a
    elif hasattr(selected, '__iter__'):
        return iter_2_array(selected)
    else:
        raise TypeError('Wrong field line selection format.')
