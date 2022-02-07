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
import flap_field_lines.accessories as acc
import flap

class Correlator:
    """
    TBD
    """

    def __init__(self,
                 data_file, 
                 save_path, 
                 selection, 
                 lines, 
                 tor_ind) -> None:
        
        f = io.open(data_file, 'rb')
        data = pickle.load(f)

        f.close()

        sel = flh.process_selection(selection)

        x, y, dx, dy = acc.return_view_xy(data)

        t = acc.return_coord(data, 'Time')
        t = flap.Coordinate(name='Time', 
                            mode=flap.CoordinateMode(equidistant=True), 
                            unit = 'Second', 
                            shape=(len(t)), 
                            start=t[0], 
                            step=t[1] - t[0], 
                            dimension_list=[0])

        dunit  = flap.Unit(name='Amplitude',unit='V')

        try:
            os.mkdir(save_path)
        except FileExistsError:
            pass

        for i in sel:
            xp, yp = acc.pixel_2_array(lines[0, i, tor_ind], lines[1, i, tor_ind], x[0], y[0], dx, dy)
            if (xp > data.shape[0]) or (xp < 0) or (yp > data.shape[1]) or (yp < 0):
                continue        
            d_ref = data.data[xp, yp, :]
            d_ref = flap.DataObject(data_array = d_ref, 
                                    data_unit = dunit, 
                                    coordinates = [t], 
                                    data_title = 'Reference', 
                                    exp_id = '123456_01')
            data_ccf = data.ccf(ref=d_ref, coordinate='Time', 
                                options={'Normalize' : True})
            data_ccf.save(save_path + f'/ccf_{tor_ind}_{i}.dat', protocol=4)
