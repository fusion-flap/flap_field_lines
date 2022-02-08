# -*- coding: utf-8 -*-
"""
Created on Fri Feb 4 2022

@author: lordofbejgli
"""

import imp
import io
import pickle
import os
from typing import Protocol
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import trapz
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
            
        self.pol_selection = []
        self.tor_selection = []
    
    def update_lines(self, surface, view, config='EIM'):
        if type(view) is not imp.ImageProjector:
            raise TypeError('View should be of type ImageProjector!íqn')

        _, self.lines = acc.get_lines(surface, ':', 'both', view, config)

    def select_contour(self, tor_ind, pol_sel):
        self.pol_selection = flh.process_selection(pol_sel)
        self.tor_selection = [tor_ind for i in range(len(self.pol_selection))]

    def select_line(self):
        pass

    def select(self):
        pass

    def correlate(self):
        for i in range(len(self.pol_selection)):
            pol, tor = self.pol_selection[i], self.tor_selection[i]
            xp, yp = acc.pixel_2_array(self.lines[0, pol, tor], 
                                       self.lines[1, pol, tor], 
                                       self.x[0], self.y[0], self.dx, self.dy)
            if (xp >= self.data.shape[0]) or (xp < 0) or (yp >= self.data.shape[1]) or (yp < 0):
                continue        
            d_ref = self.data.data[xp, yp, :]
            d_ref = flap.DataObject(data_array = d_ref, 
                                    data_unit = self.dunit, 
                                    coordinates = [self.t], 
                                    data_title = 'Reference', 
                                    exp_id = '123456_01')
            data_ccf = self.data.ccf(ref=d_ref, coordinate='Time', 
                                     options={'Normalize' : True})
            data_ccf.save(self.save_path + f'/ccf_{self.surface}_{tor}_{pol}.dat', protocol=4)
