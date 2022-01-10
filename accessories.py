# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 2021

@author: lordofbejgli
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import trapz
import flap_field_lines.field_line_handler as flh
import flap

def get_lines(surf, lines, direction, view):
    handler = flh.FieldLineHandler(configuration='EIM')
    handler.update_read_parameters(
        surfaces=surf, lines=lines, direction=direction)
    handler.load_data()
    lines = np.squeeze(handler.return_field_lines())
    lines_2 = view.calc_pixel_coord(lines)
    return lines, lines_2

def get_surfs(surfs, tor_r, direction, view):
    s_handler = flh.FieldLineHandler(configuration='EIM')
    s_handler.update_read_parameters(
        surfaces=surfs, lines=':', tor_range=tor_r, direction=direction)
    s_handler.load_data()
    surfs = np.squeeze(s_handler.return_field_lines())
    surfs_2 = view.calc_pixel_coord(surfs)
    return surfs, surfs_2

def give_plot(xlim, ylim):
    fig = plt.figure()
    plt.xlim(xlim)
    plt.ylim(ylim)
    return fig, fig.axes[0]

def stamp_surfs(tor_r, view, color='y', direction='backward'):
    _, s2 = get_surfs('10:95:10', tor_r, direction, view)
    for i in range(9):
        plt.plot(s2[0, :, i], s2[1, :, i], color)

def stamp_lines(lines_2, tor_r, color='r', axes=None):
    tor_r = flh.process_selection(tor_r)
    for i in range(lines_2.shape[1]):
        if axes is not None:
            axes.plot(lines_2[0, i, tor_r], lines_2[1, i, tor_r], color)
        else:
            plt.plot(lines_2[0, i, tor_r], lines_2[1, i, tor_r], color)

def stamp_surfs_2(surfs_2, color='y', axes=None):
    for i in range(surfs_2.shape[2]):
        if axes is not None:
            axes.plot(surfs_2[0, :, i], surfs_2[1, :, i], color)
        else:
            plt.plot(surfs_2[0, :, i], surfs_2[1, :, i], color)

def plot_frame(data, frame, x=[316, 707], y=[384, 639]):
    fig = give_plot(x,y)
    im = plt.imshow(data.data[:,:,frame].T[::-1,:], extent=[x[0], x[-1], y[0], y[-1]])
    return im, fig

def pixel_2_array(x, y, x0, y0):
    x = x - x0
    y = y-y0
    return int(np.rint(x)), int(np.rint(y))

def array_2_pixel(x, y, x0, y0):
    x = x + x0
    y = y + y0
    return x, y

def plot_corr(fig, 
              ax, 
              data, 
              vmin, 
              vmax, 
              label, 
              xref, 
              yref, 
              lines, 
              range, 
              color, 
              color_ref, 
              title, 
              surfs=None, 
              x=[316, 707], 
              y=[384, 639]):
    ax.set_xlabel('x (px)', size='small')
    ax.set_ylabel('y (px)', size='small')
    ax.tick_params(axis='both', labelsize='small')
    im = ax.imshow(data, cmap='bwr', vmin=vmin, vmax=vmax, extent=[x[0], x[-1], y[0], y[-1]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label, size='small')
    cbar.ax.tick_params(labelsize='small')
    stamp_lines(lines, range, color, ax)
    if surfs is not None:
        stamp_surfs_2(surfs, color, ax)
    ax.plot(xref, yref, c=color_ref, marker='*')
    ax.set_title(title, size='small')

def create_book(data, 
                savefile, 
                selection, 
                lines, 
                pol_r, 
                tor_r, 
                surfs, 
                tor_ind, 
                title, 
                corr_v = 1, 
                time_lag = 5, 
                color_line = 'k:', 
                color_ref = 'r'):
    pol_r = flh.process_selection(pol_r)
    tor_rp = flh.process_selection(tor_r)
    sel = flh.process_selection(selection)

    y_raw = data.coordinate('Image y', options={'Change only' : True})[0]
    y_raw = np.squeeze(y_raw)
    x_raw = data.coordinate('Image x', options={'Change only' : True})[0]
    x_raw = np.squeeze(x_raw)
    y = [x_raw[0], 1023-x_raw[0]]
    x = [y_raw[0], 1023-y_raw[0]]

    t = np.squeeze(data.coordinate('Time', options={'Change only' : True})[0])
    dt = (t[1] - t[0]) * 10**6
    t = flap.Coordinate(name='Time', 
                        mode=flap.CoordinateMode(equidistant=True), 
                        unit = 'Second', 
                        shape=(len(t)), 
                        start=t[0], 
                        step=t[1] - t[0], 
                        dimension_list=[0])

    dunit  = flap.Unit(name='Amplitude',unit='V')
    
    with PdfPages(savefile) as pdf:
        for i in sel:
            xp, yp = pixel_2_array(lines[0, i, tor_ind], lines[1, i, tor_ind], x[0], y[0])
            if (xp > data.shape[0]) or (xp < 0) or (yp > data.shape[1]) or (yp < 0):
                continue        
            fig, axes = plt.subplots(3,2, sharex=True, sharey=True)
            fig.set_size_inches(8.25, 11.75)
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
            max_p = np.argmax(data_ccf.data, axis=2)
            mid_p = int(np.floor(data_ccf.data.shape[2]) / 2)
            plot_corr(fig, 
                      axes[0,0], 
                      (max_p.T[::-1,:]-mid_p) * dt, 
                      -time_lag*dt, time_lag*dt, 
                      r'Time lag ($\mu$s)', 
                      lines[0, i, tor_ind], 
                      lines[1, i, tor_ind], 
                      lines[:, pol_r, :], 
                      tor_r, 
                      color_line, 
                      color_ref, 
                      title + ', CCF Max Offset', 
                      surfs, 
                      x, 
                      y)
            axes[0,0].plot(lines[0, i, tor_rp], lines[1, i, tor_rp], c=color_ref, ls=':')
            axes[0,0].plot(lines[0, i, 0:160], lines[1, i, 0:160], c=color_ref, ls=':')
            axes[0,0].plot(lines[0, i, 7050:7302], lines[1, i, 7050:7302], c=color_ref, ls=':')
            plot_corr(fig, 
                      axes[0,1], 
                      data_ccf.data[:,:,mid_p-6].T[::-1,:], 
                      -corr_v, 
                      corr_v, 
                      'XCorr', 
                      lines[0, i, tor_ind], 
                      lines[1, i, tor_ind], 
                      lines[:, pol_r, :], 
                      tor_r, 
                      color_line,  
                      color_ref, 
                      title + r', CCF, t = -66 $\mu$s', 
                      surfs, 
                      x, 
                      y)
            axes[0,1].plot(lines[0, i, tor_rp], lines[1, i, tor_rp], c=color_ref, ls=':')
            axes[0,1].plot(lines[0, i, 0:160], lines[1, i, 0:160], c=color_ref, ls=':')
            axes[0,1].plot(lines[0, i, 7050:7302], lines[1, i, 7050:7302], c=color_ref, ls=':')
            plot_corr(fig, 
                      axes[1,0], 
                      data_ccf.data[:,:,mid_p-3].T[::-1,:], 
                      -corr_v, 
                      corr_v, 
                      'XCorr', 
                      lines[0, i, tor_ind], 
                      lines[1, i, tor_ind], 
                      lines[:, pol_r, :], 
                      tor_r, 
                      color_line,  
                      color_ref, 
                      title + r', CCF, t = -33 $\mu$s', 
                      surfs, 
                      x, 
                      y)
            axes[1,0].plot(lines[0, i, tor_rp], lines[1, i, tor_rp], c=color_ref, ls=':')
            axes[1,0].plot(lines[0, i, 0:160], lines[1, i, 0:160], c=color_ref, ls=':')
            axes[1,0].plot(lines[0, i, 7050:7302], lines[1, i, 7050:7302], c=color_ref, ls=':')
            plot_corr(fig, 
                      axes[1,1], 
                      data_ccf.data[:,:,mid_p].T[::-1,:], 
                      -corr_v, 
                      corr_v, 
                      'XCorr', 
                      lines[0, i, tor_ind], 
                      lines[1, i, tor_ind], 
                      lines[:, pol_r, :], 
                      tor_r, 
                      color_line,  
                      color_ref, 
                      title + r', CCF, t = 0 $\mu$s', 
                      surfs, 
                      x, 
                      y)
            axes[1,1].plot(lines[0, i, tor_rp], lines[1, i, tor_rp], c=color_ref, ls=':')
            axes[1,1].plot(lines[0, i, 0:160], lines[1, i, 0:160], c=color_ref, ls=':')
            axes[1,1].plot(lines[0, i, 7050:7302], lines[1, i, 7050:7302], c=color_ref, ls=':')
            plot_corr(fig, 
                      axes[2,0], 
                      data_ccf.data[:,:,mid_p+3].T[::-1,:], 
                      -corr_v, 
                      corr_v, 
                      'XCorr', 
                      lines[0, i, tor_ind], 
                      lines[1, i, tor_ind], 
                      lines[:, pol_r, :], 
                      tor_r, 
                      color_line,  
                      color_ref, 
                      title + r', CCF, t = 33 $\mu$s', 
                      surfs, 
                      x, 
                      y)
            axes[2,0].plot(lines[0, i, tor_rp], lines[1, i, tor_rp], c=color_ref, ls=':')
            axes[2,0].plot(lines[0, i, 0:160], lines[1, i, 0:160], c=color_ref, ls=':')
            axes[2,0].plot(lines[0, i, 7050:7302], lines[1, i, 7050:7302], c=color_ref, ls=':')
            plot_corr(fig, 
                      axes[2,1], 
                      data_ccf.data[:,:,mid_p+6].T[::-1,:], 
                      -corr_v, 
                      corr_v, 
                      'XCorr', 
                      lines[0, i, tor_ind], 
                      lines[1, i, tor_ind], 
                      lines[:, pol_r, :], 
                      tor_r, 
                      color_line,  
                      color_ref, 
                      title + r', CCF, t = 66 $\mu$s', 
                      surfs, 
                      x, 
                      y)
            axes[2,1].plot(lines[0, i, tor_rp], lines[1, i, tor_rp], c=color_ref, ls=':')
            axes[2,1].plot(lines[0, i, 0:160], lines[1, i, 0:160], c=color_ref, ls=':')
            axes[2,1].plot(lines[0, i, 7050:7302], lines[1, i, 7050:7302], c=color_ref, ls=':')
            pdf.savefig()

def compare_filters(data, steep=0.2, loss=3, att=20, type='Elliptic', f_high=10000, f_low=1000):
    filter_options = {'Type' : 'Bandpass', 
                      'f_low' : f_low, 
                      'f_high' : f_high, 
                      'Tau' : 1.1111111234640703e-05, 
                      'Design' : type, 
                      'Steepness' : steep, 
                      'Loss' : loss, 
                      'Attenuation' : att}
    
    data_apsd = data.apsd(coordinate='Time', options={'Trend removal' : 'Mean'})
    f = np.squeeze(data_apsd.coordinate('Frequency', options={'Change only' : True})[0])
    data_fil = data.filter_data(coordinate='Time', options=filter_options)
    data_fil_apsd = data_fil.apsd(coordinate='Time', options={'Trend removal' : 'Mean'})
    wp = [(f_low + f_low * steep / 2), (f_high - f_high * steep / 2)]
    wp[0] = np.argmin(np.abs(f - wp[0]))
    wp[1] = np.argmin(np.abs(f - wp[1]))
    err = np.sqrt(np.sum((data_apsd.data[wp[0]:wp[1]] - data_fil_apsd.data[wp[0]:wp[1]])**2))
    plt.figure()
    plt.loglog(f, data_apsd.data)
    plt.loglog(f, data_fil_apsd.data)
    plt.title(f'steep: {steep}, loss: {loss}, attenuation: {att}, type: {type}, ' + r'$\Delta$P:' + f'{err:.3f}', size='small')

def mean_data(data):
    data_mean = np.mean(data.data, axis=2)
    t = np.squeeze(data.coordinate('Time', options={'Change only' : True})[0])
    data_all = trapz(data.data, x=t, axis=2)
    return data_mean, data_all

def data_spectral(data_apsd, roi):
    mean_spect = np.mean(np.mean(data_apsd.data, axis=0), axis=0)
    f = np.squeeze(data_apsd.coordinate('Frequency', options={'Change only' : True})[0])
    roi[0] = np.argmin(np.abs(f - roi[0]))
    roi[1] = np.argmin(np.abs(f - roi[1]))
    roi_power = trapz(data_apsd.data[:,:,roi[0]:roi[1]], x=f[roi[0]:roi[1]], axis=2)
    roi_max = np.argmax(data_apsd.data[:,:,roi[0]:roi[1]], axis=2)
    for i in roi_max.shape[0]:
        for j in roi_max.shape[1]:
            roi_max[i, j] = f[roi_max[i, j]]
    return mean_spect, roi_power, roi_max
