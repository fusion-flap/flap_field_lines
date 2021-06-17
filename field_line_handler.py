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

    def __init__(self, 
                 path='/data/W7-X/processed_data/flux_surfaces/EIM+252_detailed_w_o_limiters_w_o_torsion/field_lines', 
                 configuration='EIM', 
                 surface=None, 
                 lines=None, 
                 tor_range=None, 
                 direction='forward'):
        self.file = path + '/field_lines_tor_ang_1.85_1turn_%s+252_w_o_limiters_w_o_torsion_w_characteristics_surf_0'
        if configuration in ('EIM', 'FTM'):
            self.file = self.file % configuration + '%s.sav'
        else:
            raise ValueError('Wrong configuration.')

        if direction in ('forward', 'backward', 'both'):
            self.direction = direction
        else:
            raise ValueError('Direction of field lines is invalid. Should be "forward", "backward" or "both".\n')
        self.__field_lines = []

        selected_surf = range(200)
        if surface is not None:
            selected_surf = process_selection(surface)

        selected_lines = None
        if lines is not None:
            selected_lines = process_selection(lines)
        
        selected_range = None
        if tor_range is not None:
            selected_range = process_selection(tor_range)

        for i in selected_surf:
            try:
                self.__field_lines = self.return_lines_from_surf(i, selected_lines, selected_range)
                break
            except NoSurfaceFileError as err:
                if surface is not None:
                    print(err)    
        
        if self.__field_lines == []:
            raise IOError("No flux surface files found.\n")

        i = selected_surf.index(i)
        if len(selected_surf) > i + 1:
            self.__field_lines = self.__field_lines[..., np.newaxis]

            for j in selected_surf[i + 1:]:
                try:
                    new_lines = self.return_lines_from_surf(j, selected_lines, selected_range)
                except NoSurfaceFileError as err:
                    if surface is not None:
                        print(err)
                    continue
                self.__field_lines = np.concatenate((self.__field_lines, 
                                                   new_lines[..., np.newaxis]), 
                                                   axis=-1)    

    def return_lines_from_surf(self, surf_no, lines_no, tor_range):
        string_no = '0%d' % surf_no
        if surf_no < 10:
            string_no = '0' + string_no
        
        if not os.path.isfile(self.file % string_no):
            raise NoSurfaceFileError(surf_no)

        surf = readsav(self.file % string_no)
        lines = []

        if lines_no is None:
            lines_no = range(len(surf['surface'][0][4]))

        if self.direction == 'forward':
            lines = np.array([surf['surface'][0][4][lines_no], 
                              surf['surface'][0][5][lines_no], 
                              surf['surface'][0][6][lines_no]])
        elif self.direction == 'backward':
            lines = np.array([surf['surface'][0][7][lines_no], 
                              surf['surface'][0][8][lines_no], 
                              surf['surface'][0][9][lines_no]])
        elif self.direction == 'both':
            lines = np.array([surf['surface'][0][4][lines_no], surf['surface'][0][5][lines_no], surf['surface'][0][6][lines_no]])
            lines = np.concatenate(np.array([surf['surface'][0][7][lines_no, -1::-1], 
                                             surf['surface'][0][8][lines_no, -1::-1], 
                                             surf['surface'][0][9][lines_no, -1::-1]]), 
                                             lines, axis=2)
        
        if tor_range is None:
            tor_range = range(len(surf['surface'][0][4][0]))

        return lines[:, :, tor_range]

    def get_field_lines(self):
        return self.__field_lines
                

def process_selection(selected):
    if isinstance(selected, int):
        return [selected]
    elif isinstance(selected, str):
        selected = selected.split(':')
        first = int(selected[0])
        last = int(selected[1])
        if len(selected) == 3:
            step = int(selected[2])
            return range(first, last, step)
        elif len(selected) > 3:
            raise ValueError('Input string should be of format from:to(:by).')
        else:
            return range(first, last)
    elif hasattr(selected, '__iter__'):
        a = []
        for x in selected:
            if isinstance(x, int):
                a.append(x)
            else:
                raise TypeError('Field line selection should only have integers.')
        return a
    else:
        raise TypeError('Wrong field line selection format.')
