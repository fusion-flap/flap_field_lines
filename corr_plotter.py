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

    def __init__(self, corr_file, ref_tor = 3625, save_path=None) -> None:
        f = io.open(corr_file, 'rb')
        
        self.data = pickle.load(f)

        f.close()

        if save_path is not None:
            self.save_path = save_path
            try:
                os.mkdir(self.save_path)
            except FileExistsError:
                pass
        else:
            self.save_path = os.path.dirname(corr_file)

        _, self.lines = acc.get_lines(self.data.info['surface'], 
                                      ':', 'both', 
                                      self.data.info['view'], 
                                      self.data.info['config'])
        _, self.surfs = acc.get_surfs(self.data.info['surface'], 
                                      ref_tor,
                                      'both', 
                                      self.data.info['view'], 
                                      self.data.info['config'])

    def create_slides(self,
                      savefile, 
                    pol_r, 
                    tor_r, 
                    surfs, 
                    tor_ind, 
                    title, 
                    corr_v = 1, 
                    color_line = 'k', 
                    color_ref = 'r'):

        savefile = self.save_path + '/' + savefile

        with PdfPages(savefile) as pdf:
            for i in range(self.data.shape[3]): 
                fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
                fig.set_size_inches(10.08, 7.56)
                axes[0,0].set_ylim(y[0], y[-1])
                axes[0,0].set_xlim(x[0], x[-1])
                mid_p = int(np.floor(self.data.shape[2]) / 2)
                acc.plot_corr(fig, 
                        axes.ravel()[0], 
                        self.data.data[:,:,mid_p-4].T[::-1,:], 
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