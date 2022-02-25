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

    def __init__(self, corr_file, plot_surfs=None, ref_tor=3425, save_path=None) -> None:
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

        self.save_title = os.path.basename(corr_file).split('.')[0]

        _, self.lines = acc.get_lines(self.data.info['surface'], 
                                      ':', 'both', 
                                      self.data.info['view'], 
                                      self.data.info['config'])
        if plot_surfs is not None:
            _, self.surfs = acc.get_surfs(plot_surfs, 
                                        ref_tor,
                                        'both', 
                                        self.data.info['view'], 
                                        self.data.info['config'])
        else:
            self.surfs = None

    def create_slides(self,
                      pol_r, 
                      tor_r, 
                      corr_v = 1, 
                      color_line = 'k', 
                      color_ref = 'r'):

        savefile = self.save_path + '/ ' + self.save_title + '.pdf'

        x, y, _, _ = acc.return_view_xy(self.data)

        ref_pol = acc.return_coord(self.data, 'Poloidal coordinate')
        ref_tor = acc.return_coord(self.data, 'Toroidal coordinate')

        pol_r = flh.process_selection(pol_r)
        tor_r_pro = flh.process_selection(tor_r)

        title = self.data.exp_id + ', ' + \
                self.data.info['config'] + ', ' + \
                self.data.info['view'].viewpoint.split('-')[1]

        with PdfPages(savefile) as pdf:
            for i in range(len(ref_tor)): 
                mid_p = int(np.floor(self.data.shape[2]) / 2)
                fig, axes = self.make_slide_layout(x, y)
                fig.suptitle(f'Field Line: {ref_pol[i]}')
                t_lags = [-4, -2, 0, 4]
                for j in range(4):
                    t_j = t_lags[j]
                    acc.plot_corr(fig, 
                                  axes.ravel()[j], 
                                  self.data.data[:,:,mid_p + t_j, i], 
                                  -corr_v, 
                                  corr_v, 
                                  'XCorr', 
                                  self.lines[0, ref_pol[i], ref_tor[i]], 
                                  self.lines[1, ref_pol[i], ref_tor[i]], 
                                  self.lines[:, pol_r, :], 
                                  tor_r, 
                                  color_line,  
                                  color_ref, 
                                  title + f', CCF, t = {t_j*11}' +  r'$\mu$s', 
                                  self.surfs, 
                                  x, 
                                  y)
                    acc.plot_ref_line(axes.ravel()[j], self.lines, ref_pol[i], tor_r_pro, color_ref)
                pdf.savefig()
                plt.close()
                fig, axes = self.make_slide_layout(x, y)
                fig.suptitle(f'Field Line: {ref_pol[i]}')
                t_lags = [-4, 0, 2, 4]
                for j in range(4):
                    t_j = t_lags[j]
                    acc.plot_corr(fig, 
                                  axes.ravel()[j], 
                                  self.data.data[:,:,mid_p + t_j, i], 
                                  -corr_v, 
                                  corr_v, 
                                  'XCorr', 
                                  self.lines[0, ref_pol[i], ref_tor[i]], 
                                  self.lines[1, ref_pol[i], ref_tor[i]], 
                                  self.lines[:, pol_r, :], 
                                  tor_r, 
                                  color_line,  
                                  color_ref, 
                                  title + f', CCF, t = {t_j*11}' +  r'$\mu$s', 
                                  self.surfs, 
                                  x, 
                                  y)
                    acc.plot_ref_line(axes.ravel()[j], self.lines, ref_pol[i], tor_r_pro, color_ref)
                pdf.savefig()
                plt.close()

    def make_slide_layout(self, x, y):
        fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
        fig.set_size_inches(10.08, 7.56)
        axes[0,0].set_ylim(y[0], y[-1])
        axes[0,0].set_xlim(x[0], x[-1])
        return fig, axes
    
    def calc_surf_contour(self, surf, tor):
        _, surf = acc.get_surfs(surf, tor, 'both', self.data.info['view'], self.data.info['config'])
        x, y, dx, dy = acc.return_view_xy(self.data)
        xp, yp = acc.pixel_2_array(surf[0, :],surf[1, :], x, y, dx, dy)
        pol = [p for i, j, p in zip(xp, yp, range(len(xp))) if not acc.is_out_of_frame(self.data, i, j)]
        xp = [[i, j] for i, j in zip(xp, yp) if not acc.is_out_of_frame(self.data, i, j)]
        xp = acc.unique_list(xp)
        yp = np.array(xp)[:,1]
        xp = np.array(xp)[:,0]
        return xp, yp, pol
        

    def make_slides_t_lag(self,
                      pol_r, 
                      tor_r, 
                      corr_v = 1, 
                      color_line = 'k', 
                      color_ref = 'r'):

        savefile = self.save_path + '/ ' + self.save_title + '_t_lag.pdf'

        x, y, _, _ = acc.return_view_xy(self.data)

        ref_pol = acc.return_coord(self.data, 'Poloidal coordinate')
        ref_tor = acc.return_coord(self.data, 'Toroidal coordinate')

        pol_r = flh.process_selection(pol_r)
        tor_r_pro = flh.process_selection(tor_r)
        
        t = acc.return_coord(self.data, 'Time lag')
        dt = (t[1] - t[0]) * 10**6

        title = self.data.exp_id + ', ' + \
                self.data.info['config'] + ', ' + \
                self.data.info['view'].viewpoint.split('-')[1]

        with PdfPages(savefile) as pdf:
            for i in range(len(ref_tor)): 
                max_p = t[np.argmax(self.data.data[:,:,:,i], axis=2)] * 10**6
                fig, axes = self.make_slide_layout(x, y)
                fig.suptitle(f'Field Line: {ref_pol[i]}')
                t_lags = [3, 7]
                for j in range(2):
                    t_j = t_lags[j]
                    acc.plot_corr(fig, 
                                  axes.ravel()[j], 
                                  max_p, 
                                  -dt * t_j, 
                                  dt * t_j, 
                                  r'Time lag ($\mu$s)', 
                                  self.lines[0, ref_pol[i], ref_tor[i]], 
                                  self.lines[1, ref_pol[i], ref_tor[i]], 
                                  self.lines[:, pol_r, :], 
                                  tor_r, 
                                  color_line,  
                                  color_ref, 
                                  title + ', CCF Max Offset', 
                                  self.surfs, 
                                  x, 
                                  y)
                    acc.plot_ref_line(axes.ravel()[j], self.lines, ref_pol[i], tor_r_pro, color_ref)
                
                max_v = np.amax(self.data.data[:,:,:,i], axis=2)
                acc.plot_corr(fig, 
                              axes.ravel()[2], 
                              max_v, 
                              -corr_v, 
                              corr_v, 
                              'XCorr Max', 
                              self.lines[0, ref_pol[i], ref_tor[i]], 
                              self.lines[1, ref_pol[i], ref_tor[i]], 
                              self.lines[:, pol_r, :], 
                              tor_r, 
                              color_line,  
                              color_ref, 
                              title + ', Max CCF', 
                              self.surfs, 
                              x, 
                              y)
                acc.plot_ref_line(axes.ravel()[2], self.lines, ref_pol[i], tor_r_pro, color_ref)
                pdf.savefig()
                plt.close()
