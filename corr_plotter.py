# -*- coding: utf-8 -*-
"""
Created on Fri Feb 9 2022

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

class CorrPlotter:

    def __init__(self, corr_file, view, save_path) -> None:
        pass

    def create_slides(self,
                      data, 
                      savefile, 
                    selection, 
                    lines, 
                    pol_r, 
                    tor_r, 
                    surfs, 
                    tor_ind, 
                    title, 
                    corr_v = 1, 
                    color_line = 'k', 
                    color_ref = 'r'):

        x, y, dx, dy = return_view_xy(data)
        
        with PdfPages(savefile) as pdf:
            for i in sel:
                xp, yp = pixel_2_array(lines[0, i, tor_ind], lines[1, i, tor_ind], x[0], y[0], dx, dy)
                if (xp > data.shape[0]) or (xp < 0) or (yp > data.shape[1]) or (yp < 0):
                    continue        
                fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
                fig.set_size_inches(10.08, 7.56)
                axes[0,0].set_ylim(y[0], y[-1])
                axes[0,0].set_xlim(x[0], x[-1])
                d_ref = data.data[xp, yp, :]
                d_ref = flap.DataObject(data_array = d_ref, 
                                        data_unit = dunit, 
                                        coordinates = [t], 
                                        data_title = 'Reference', 
                                        exp_id = '123456_01')
                data_ccf = data.ccf(ref=d_ref, coordinate='Time', 
                                    options={'Normalize' : True})
                mid_p = int(np.floor(data_ccf.data.shape[2]) / 2)
                plot_corr(fig, 
                        axes.ravel()[0], 
                        data_ccf.data[:,:,mid_p-4].T[::-1,:], 
                        -corr_v, 
                        corr_v, 
                        'XCorr', 
                        lines[0, i, tor_ind], 
                        lines[1, i, tor_ind], 
                        lines[:, pol_r, :], 
                        tor_r, 
                        color_line,  
                        color_ref, 
                        title + f', CCF, t = -44' +  r'$\mu$s', 
                        surfs, 
                        x, 
                        y)
                plot_ref_line(axes.ravel()[0], lines, i, tor_rp, color_ref)
                for j in range(3):
                    plot_corr(fig, 
                            axes.ravel()[j + 1], 
                            data_ccf.data[:,:,mid_p+2*j].T[::-1,:], 
                            -corr_v, 
                            corr_v, 
                            'XCorr', 
                            lines[0, i, tor_ind], 
                            lines[1, i, tor_ind], 
                            lines[:, pol_r, :], 
                            tor_r, 
                            color_line,  
                            color_ref, 
                            title + f', CCF, t = {j*22}' +  r'$\mu$s', 
                            surfs, 
                            x, 
                            y)
                    plot_ref_line(axes.ravel()[j + 1], lines, i, tor_rp, color_ref)
                pdf.savefig()