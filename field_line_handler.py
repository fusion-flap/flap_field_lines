# -*- coding: utf-8 -*-
"""
Created on Tue May 31 2021

@author: lordofbejgli
"""

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
    def __init__(self, 
                 path='/data/W7-X/processed_data/flux_surfaces/%s+252_detailed_w_o_limiters_w_o_torsion/field_lines', 
                 configuration='EIM', 
                 surface=None, 
                 lines=None, 
                 tor_range=None, 
                 direction='forward',
                 getB = False,
                 gradB = False):
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

        #processes selections
        selected_surf = range(200)
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

    def get_field_lines(self):
        """
        Returnes stored data
        """
        return self.__field_lines
                

    def get_B(self):
        """
        Returnes stored data
        """
        return self.__B

    def get_gradB(self):
        """
        Returnes stored data
        """
        return self.__gradB

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
