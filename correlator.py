# -*- coding: utf-8 -*-
"""
Created on Fri Feb 4 2022

@author: lordofbejgli
"""

import imp
import io
import pickle
import os
import numpy as np
import flap_field_lines.field_line_handler as flh
import flap_field_lines.image_projector as imp
import flap_field_lines.accessories as acc
import flap

class Correlator:
    """
    TBD
    """

    def __init__(self,
                 data_file, 
                 save_path,
                 surface=None,
                 view=None,
                 config=None) -> None:
        
        f = io.open(data_file, 'rb')
        self.data = pickle.load(f)

        f.close()

        self.x, self.y, self.dx, self.dy = acc.return_view_xy(self.data)

        t = acc.return_coord(self.data, 'Time')
        self.t = flap.Coordinate(name='Time', 
                            mode=flap.CoordinateMode(equidistant=True), 
                            unit = 'Second', 
                            shape=(len(t)), 
                            start=t[0], 
                            step=t[1] - t[0], 
                            dimension_list=[0])

        self.dunit  = flap.Unit(name='Amplitude',unit='V')

        self.save_path = save_path
        try:
            os.mkdir(self.save_path)
        except FileExistsError:
            pass
        
        if surface is not None:
            self.update_lines(surface, view, config)
        else:
            self.surface = None
            self.lines = None

        self.selection = []
        self.selection_type = None

    def update_lines(self, surface, view, config='EIM'):
        if type(view) is not imp.ImageProjector:
            raise TypeError('View should be of type ImageProjector!íqn')

        _, self.lines = acc.get_lines(surface, ':', 'both', view, config)
        self.surface = surface

    def select_contour(self, tor_ind, pol_sel):
        pol_sel = flh.process_selection(pol_sel)
        self.selection = [[i, tor_ind] for i in pol_sel]
        self.remove_out_of_frame()
        self.selection_type = 'tor'

    def select_line(self, tor_sel, pol_ind):
        tor_sel = flh.process_selection(tor_sel)
        self.selection = [[pol_ind, i] for i in tor_sel]
        self.remove_out_of_frame()
        self.selection_type = 'pol'

    def select(self, tor_sel, pol_sel):
        tor_sel = flh.process_selection(tor_sel)
        pol_sel = flh.process_selection(pol_sel)
        if len(tor_sel) != len(pol_sel):
            raise ValueError('Numer of given poloidal coordinates does not \
                              match the number of toroidal coordinates.\n')
        self.selection = [[pol_sel[i], tor_sel[i]] for i in range(len(tor_sel))]
        self.remove_out_of_frame()
        self.selection_type = 'mix'

    def remove_out_of_frame(self):
        for sel in self.selection:
            xp, yp = acc.pixel_2_array(self.lines[0, sel[0], sel[1]], 
                                       self.lines[1, sel[0], sel[1]], 
                                       self.x[0], self.y[0], self.dx, self.dy)
            if (xp >= self.data.shape[0]) or (xp < 0) or (yp >= self.data.shape[1]) or (yp < 0):
                self.selection.remove(sel)

    def return_corr(self, i):
        pol, tor = self.selection[i][0], self.selection[i][1]
        xp, yp = acc.pixel_2_array(self.lines[0, pol, tor], 
                                   self.lines[1, pol, tor], 
                                   self.x[0], self.y[0], self.dx, self.dy)
                
        d_ref = self.data.data[xp, yp, :]
        d_ref = flap.DataObject(data_array = d_ref, 
                                data_unit = self.dunit, 
                                coordinates = [self.t], 
                                data_title = 'Reference', 
                                exp_id = '123456_01')
        data_ccf = self.data.ccf(ref=d_ref, coordinate='Time', 
                                 options={'Normalize' : True})
        t = acc.return_coord(data_ccf, 'Time lag')
        mid_p = int(np.floor(len(t)) / 2)
        data_ccf = data_ccf.slice_data(slicing={'Time lag' : t[mid_p-15:mid_p+16]})
        return data_ccf

    def correlate(self):
        data_ccf = self.return_corr(0)
        
        t = data_ccf.get_coordinate_object('Time lag')
        x = data_ccf.get_coordinate_object('Image x')
        y = data_ccf.get_coordinate_object('Image y')
        dunit = data_ccf.data_unit

        ccf_arrays = data_ccf.data
        if len(self.selection) > 1:
            ccf_arrays = ccf_arrays[..., np.newaxis]
        else:
            pass

        for i in range(1, len(self.pol_selection)):
            data_ccf = self.return_corr(i)
            ccf_arrays = np.concatenate((ccf_arrays, 
                                         data_ccf.data[..., np.newaxis]), 
                                         axis = -1)
        
        ref_coord_id = ""

        if self.selection_type == 'tor':
            ref_coord = flap.Coordinate(name='Poloidal coordinate', 
                                        mode=flap.CoordinateMode(equidistant=False), 
                                        unit = 'deg', 
                                        shape=(len(self.selection)), 
                                        start=self.selection[0][0], 
                                        values= [sel[0] for sel in self.selection], 
                                        dimension_list=[3])
            ref_coord_id = f'_{self.selection[0][1]}'
        elif self.selection_type == 'pol':
            ref_coord = flap.Coordinate(name='Toroidal coordinate', 
                                        mode=flap.CoordinateMode(equidistant=False), 
                                        unit = 'bin', 
                                        shape=(len(self.selection)), 
                                        start=self.selection[0][1],
                                        values= [sel[1] for sel in self.selection], 
                                        dimension_list=[3])
            ref_coord_id = f'_{self.selection[0][0]}'
        elif self.selection_type == 'mix':
            ref_coord = flap.Coordinate(name='Reference point order', 
                                        mode=flap.CoordinateMode(equidistant=True), 
                                        unit = 'bin', 
                                        shape=(len(self.selection)), 
                                        start=0,
                                        values= [i for i in range(len(self.selection))], 
                                        dimension_list=[3])

        data_ccf = flap.DataObject(data_array = ccf_arrays, 
                                   data_unit = dunit, 
                                   coordinates = [t, x, y, ref_coord])

        data_ccf.save(self.save_path + f'/ccf_{self.surface}_{self.selection_type}{ref_coord_id}.dat', protocol=4)
