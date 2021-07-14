# -*- coding: utf-8 -*-
"""
Created on Tue May 31 2021

@author: lordofbejgli
"""

from typing import get_args
import numpy as np
import os

from scipy.io import readsav
from errors import *

class FieldLineHandler:
    """
    This class reads and stores field lines of various flux surfaces from the 
    given .sav files. In its current form, it's equipped to handle W7X field 
    lines. Onew .sav file contains the forward and backward calculated field 
    lines of one flux surface. Typically there's 360 field line per surface, 
    which are 3651 bin long. Results are stored in an numpy array. It can be 
    accessed via the get_field_lines() function.

    Stored array may have 2, 3 or 4 dimensions. Thes are:
        - 1st dimension: 3 long, x, y, z coordinates
        - last dimension: flux surface coordinate if there's more
        - 2nd and 3rd: poloidal (if multiple lines are selected per surface) 
          and toroidal coordinates
    """
    __valid_configs = ('EIM', 'FTM', 'KJM001')

    def __init__(self, 
                 path=None, 
                 configuration=None):
        """
        TBD
        """
        if path is None:
            self.path = '/data/W7-X/processed_data/flux_surfaces/%s+252_detailed_w_o_limiters_w_o_torsion'
            if configuration in self.__valid_configs:
                try:
                    self.path = self.path % configuration 
                except TypeError:
                    pass
                fs_info = self.path + '/fs_info.sav'
                if not os.path.isfile(fs_info):
                    raise NoFsInfoError
            else:
                raise WrongConfigurationError(configuration)
        else:
            self.path = path.replace('/fs_info.sav', '')
            fs_info = path
        self.configuration = configuration
        self.__fs_info = self.__read_fs_info(fs_info)
        self.__field_lines = []
        self.__B = None
        self.__gradB = None
        self.surfaces = []
        self.surface_files = []
        self.read_files = []
        self.lines = None
        self.tor_range = None
        self.direction = None

    def __read_fs_info(self, file):
        fs_info = readsav(file)
        needed_info = {}
        needed_info['iota'] = fs_info['fs_info'][0][3]
        needed_info['reff'] = fs_info['fs_info'][0][4]
        needed_info['separatrix'] = fs_info['fs_info'][0][6]
        needed_info['names'] = fs_info['fs_info'][0][7]
        needed_info['flags'] = fs_info['fs_info'][0][8]
        return needed_info

    def update_read_parameters(self, 
                 path=None, 
                 surfaces=None, 
                 lines=None, 
                 tor_range=None, 
                 direction='forward'):

        if direction in ('forward', 'backward', 'both'):
            if direction != self.direction:
                self.read_files = [False for i in range(len(self.read_files))]
            self.direction = direction
        else:
            raise WrongDirectionError

        if path:
            self.path = path
        else:
            if os.path.exists(self.path + '/field_lines'):
                self.path += '/field_lines'

        if surfaces:
            try:
                surfaces = process_selection(surfaces)
            except TypeError:
                if not os.path.isfile(surfaces[0]):
                    raise
            else:
                surf_files, surfaces = self.__create_surf_file_list(surfaces)
        elif not self.surfaces:
            surf_files, surfaces = self.__create_surf_file_list(range(len(self.__fs_info['iota'])))

        self.surfaces += surfaces
        self.surface_files += surf_files
        self.read_files += [False for i in range(len(surfaces))]

        if lines:
            lines = process_selection(lines)
            if lines != self.lines:
                self.lines = lines
                self.read_files += [False for i in range(len(self.surfaces))]

        if tor_range:
            tor_range = process_selection(tor_range)
            if tor_range != self.tor_range:
                self.tor_range = tor_range
                self.read_files += [False for i in range(len(self.surfaces))]

    def load_data(self, getB=False, getGradB=False):
        pass

    def __create_surf_file_list(self, surfs):
        file = self.path + '/field_lines_tor_ang_1.85_1turn_%s+252_w_o_limiters_w_o_torsion_w_characteristics_surf_0'
        file = file % self.configuration + '%s.sav'
        file_list = []
        surf_list = []

        for i in surfs:
            string_no = str(i)
            if i < 10:
                string_no = '0' + string_no
            if os.path.isfile(file % string_no):
                file_list.append(file % string_no)
                surf_list.append(i)
        
        return file_list, surf_list

    def __read_surf_files(self, file_list, get_fl=False, get_B=False, get_gradB=False):
        surf = readsav(file_list[0])
         #if no lines are specified, chooses all
        if self.lines is None:
            self.lines = range(len(surf['surface'][0][4]))

        #if no toroidal range is specified, chooses all
        if self.tor_range is None:
            self.tor_range = range(len(surf['surface'][0][4][0]))

        if get_fl:
            field_lines = self.__extract_data_from_surf(surf, 4)

        if get_B:
            B = self.__extract_data_from_surf(surf, 10)

        if get_gradB:
            gradB = self.__extract_data_from_surf(surf, 16)
        
        if len(file_list) > 1:
            if get_fl:
                field_lines = field_lines[..., np.newaxis]
            if get_B:
                B = B[..., np.newaxis]
            if get_gradB:
                gradB = gradB[..., np.newaxis]
        else:
            return field_lines, B, gradB

        for file in file_list[1:]:
            surf = readsav(file)

            if get_fl:
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





    def __something_else(self, 
                 path='/data/W7-X/processed_data/flux_surfaces/%s+252_detailed_w_o_limiters_w_o_torsion/field_lines', 
                 configuration='EIM', 
                 surface=None, 
                 lines=None, 
                 tor_range=None, 
                 direction='forward',
                 getB=False,
                 gradB=False):
        """
        Constructor of the class. Its inputs are:
        path: Where the precalculated field lines are stored.
        configuration: Magnetic configuration of the device. Currently "EIM" 
                       and "FTM" are valid options. Raises ValueError otherwise.
        surface: Which flux surface to choose. Applicable formats:
                    - single int, for single surface selection
                    - any iterable, containing only ints
                    - string for range selection in the following format:
                      "from:to(:by)"
        lines: Which lines to select. Same format as "surface". Typically there's 
               360 lines calculated on a flux surface.
        tor_range: Toroidal range, meaning which part of the field lines to 
                   select. Typically a field line is 3651 point long. Same 
                   format as "surface".
        direction: Wether to read the field lines worward, backward or both 
                   direction. Viable options are "forward", "backward", "both". 
                   If "both" are selected, the backward calculated lines are 
                   reversed and concatenated by the forward calculated lines.
        
        The method checks input, processes the selection ranges than proceeds 
        to read the first flux surface files. If no readable file is found, 
        raises IOError. If there are more surfaces to read, extends the array 
        after the last dimension and joines the others to it.
        
        Raises: ValueError, IOError, TypeError
        """
        #checks input
        self.get_B = getB
        self.get_gradB = gradB
        self.file = path
        if configuration in ('EIM', 'FTM', 'KJM001'):
            try:
                self.file = self.file % configuration 
            except TypeError:
                pass
            fs_info = self.file + '/fs_info.sav'
            self.file += '/field_lines_tor_ang_1.85_1turn_%s+252_w_o_limiters_w_o_torsion_w_characteristics_surf_0'
            self.file = self.file % configuration + '%s.sav'
        else:
            raise ValueError('Wrong configuration.')

        if direction in ('forward', 'backward', 'both'):
            self.direction = direction
        else:
            raise ValueError('Direction of field lines is invalid. Should be "forward", "backward" or "both".')
        self.__field_lines = []
        self.__B = None
        self.__gradB = None

        fs_info = self.__read_fs_info(fs_info)
        self.__fs_info = fs_info.copy()
        self.__fs_info['iota'] = []
        self.__fs_info['reff'] = []
        self.__fs_info['flags'] = []

        #processes selections
        selected_surf = range(len(fs_info['iota']))
        if surface is not None:
            selected_surf = process_selection(surface)

        selected_lines = None
        if lines is not None:
            selected_lines = process_selection(lines)
        
        selected_range = None
        if tor_range is not None:
            selected_range = process_selection(tor_range)

        #finds and readds first readable surface file
        for i in selected_surf:
            try:
                self.__field_lines, self.__B, self.__gradB = self.__return_lines_from_surf(i, selected_lines, selected_range)
                break
            except NoSurfaceFileError as err:
                #if a user specified surface file is not found, prints error
                if surface is not None:
                    print(err)    
        
        if self.__field_lines == []:
            raise IOError("No flux surface files found.")

        self.__fs_info['iota'].append(fs_info['iota'][i])
        self.__fs_info['reff'].append(fs_info['reff'][i])
        self.__fs_info['flags'].append(fs_info['flags'][i])

        #proceeds to read the rest
        i = selected_surf.index(i)
        if len(selected_surf) > i + 1:
            #if there's more to read, extends array dimensions and  joins them 
            #by the last
            self.__field_lines = self.__field_lines[..., np.newaxis]
            if self.get_B:
                self.__B = self.__B[..., np.newaxis]
            if self.get_gradB:
                self.__gradB = self.__gradB[..., np.newaxis]

            for j in selected_surf[i + 1:]:
                try:
                    new_lines, new_B, new_gradB = self.__return_lines_from_surf(j, selected_lines, selected_range)
                except NoSurfaceFileError as err:
                    #if a user specified surface file is not found, prints error
                    if surface is not None:
                        print(err)
                    continue
                self.__field_lines = np.concatenate((self.__field_lines, 
                                                   new_lines[..., np.newaxis]), 
                                                   axis=-1)    

                self.__fs_info['iota'].append(fs_info['iota'][j])
                self.__fs_info['reff'].append(fs_info['reff'][j])
                self.__fs_info['flags'].append(fs_info['flags'][j])

                if self.get_B:
                    self.__B = np.concatenate((self.__B, 
                                            new_B[..., np.newaxis]), 
                                            axis=-1)    
                if self.get_gradB:
                    self.__gradB = np.concatenate((self.__gradB, 
                                                new_gradB[..., np.newaxis]), 
                                                axis=-1)    
            if self.__field_lines.shape[3] == 1:
                self.__field_lines = np.squeeze(self.__field_lines)
                if self.get_B:
                    self.__B = np.squeeze(self.__B)
                if self.get_gradB:
                    self.__gradB = np.squeeze(self.__gradB)

    def __return_lines_from_surf(self, surf_no, lines_no, tor_range):
        """
        Returns requested data from a flux surface file. Acts as part of the 
        constructor, not used by itself. Takes the surface number and the 
        processed selections as input.
        Raises "NoSurfaceFileError" if file not found.
        """
        #converts surface number to file number
        string_no = '%d' % surf_no
        if surf_no < 10:
            string_no = '0' + string_no
        
        #if requested file doesn't exists, raises exception
        if not os.path.isfile(self.file % string_no):
            raise NoSurfaceFileError(surf_no)

        #reads file, initializes storage variable
        surf = readsav(self.file % string_no)

        #if no lines are specified, chooses all
        if lines_no is None:
            lines_no = range(len(surf['surface'][0][4]))

        #if no toroidal range is specified, chooses all
        if tor_range is None:
            tor_range = range(len(surf['surface'][0][4][0]))

        lines = self.extract_roi_from_lines(surf, lines_no, tor_range, 4)
        if self.get_B:
            B = self.extract_roi_from_lines(surf, lines_no, tor_range, 10)
        else:
            B = None
        if self.get_gradB:
            gradB = self.extract_roi_from_lines(surf, lines_no, tor_range, 16)
        else:
            gradB = None
        return lines, B, gradB

    def extract_roi_from_lines(self, surf, lines_no, tor_range, index_no):
        lines = []
        if self.direction == 'forward':
            #reads forward calculated field lines
            lines = np.array([surf['surface'][0][index_no][lines_no], 
                              surf['surface'][0][index_no + 1][lines_no], 
                              surf['surface'][0][index_no + 2][lines_no]])
        elif self.direction == 'backward':
            #reads backward calculated field lines
            lines = np.array([surf['surface'][0][index_no + 3][lines_no], 
                              surf['surface'][0][index_no + 4][lines_no], 
                              surf['surface'][0][index_no + 5][lines_no]])
        elif self.direction == 'both':
            #reads both. backward lines are erversed and placed in front of 
            #forward lines
            lines = np.array([surf['surface'][0][index_no][lines_no], 
                              surf['surface'][0][index_no + 1][lines_no], 
                              surf['surface'][0][index_no + 2][lines_no]])
            lines = np.concatenate((np.array([surf['surface'][0][index_no + 3][lines_no, -1::-1], 
                                              surf['surface'][0][index_no + 4][lines_no, -1::-1], 
                                              surf['surface'][0][index_no + 5][lines_no, -1::-1]]), 
                                              lines), axis=2)
        return lines[:, :, tor_range]

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

def process_selection(selected):
    """
    Processes selection. Either returnes a list in case of single or mutiple 
    specific selection, or a range if a range is specified.
    Raises ValueError for wrong string format and TypeError if non-ints are given, 
    or something unexpected.
    """
    if isinstance(selected, int):
        #returns 1 long list for singe selection
        return [selected]
    elif isinstance(selected, str):
        #returns range if a string is specified. string is broken up by ":"
        selected = selected.split(':')
        first = int(selected[0])
        last = int(selected[1])
        if len(selected) == 3:
            #step bw elements is optional
            step = int(selected[2])
            return range(first, last, step)
        elif len(selected) > 3:
            raise ValueError('Input string should be of format from:to(:by).')
        else:
            return range(first, last)
    elif hasattr(selected, '__iter__'):
        #if selection is iterable, put all elements into a list
        a = []
        for x in selected:
            if isinstance(x, int):
                a.append(x)
            else:
                raise TypeError('Field line selection should only have integers.')
        return a
    else:
        raise TypeError('Wrong field line selection format.')
