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
import flap_w7x_camera

def get_lines(surf, lines, direction, view, config='EIM'):
    handler = flh.FieldLineHandler(configuration=config)
    handler.update_read_parameters(
        surfaces=surf, lines=lines, direction=direction)
    handler.load_data()
    lines = np.squeeze(handler.return_field_lines())
    lines_2 = view.calc_pixel_coord(lines)
    return lines, lines_2

def get_surfs(surfs, tor_r, direction, view, config='EIM'):
    s_handler = flh.FieldLineHandler(configuration=config)
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

def stamp_surfs(surf_select, tor_r, view, color='y', direction='backward', config='EIM'):
    _, s2 = get_surfs(surf_select, tor_r, direction, view, config)
    if len(s2.shape) < 3:
        plt.plot(s2[0, :], s2[1, :], color)
    else:
        for i in range(9):
            plt.plot(s2[0, :, i], s2[1, :, i], color)

def stamp_lines(lines_2, tor_r, color='r', axes=None):
    tor_r = flh.process_selection(tor_r)
    for i in range(lines_2.shape[1]):
        if axes is not None:
            axes.plot(lines_2[0, i, tor_r], lines_2[1, i, tor_r], color)
        else:
            plt.plot(lines_2[0, i, tor_r], lines_2[1, i, tor_r], color)

def stamp_line(line, tor_r, color='r', axes=None):
    tor_r = flh.process_selection(tor_r)
    if axes is not None:
        axes.plot(line[0, tor_r], line[1, tor_r], color)
    else:
        plt.plot(line[0, tor_r], line[1, tor_r], color)

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

def pixel_2_array(x, y, x0, y0, dx=1, dy=1):
    x = x - x0
    y = y-y0
    return int(np.floor(x / dx)), int(np.floor(y / dy))

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
    stamp_lines(lines, range, color + ':', ax)
    if surfs is not None:
        stamp_surfs_2(surfs, color, ax)
    ax.plot(xref, yref, c=color_ref, marker='*')
    ax.set_title(title, size='small')

def return_coord(data, coord):
    t = np.squeeze(data.coordinate(coord, options={'Change only' : True})[0])
    return t

def return_view_xy(data):
    y_raw = return_coord(data, 'Image y')
    x_raw = return_coord(data, 'Image x')
    y = [x_raw[0], 1023-x_raw[0]]
    x = [y_raw[0], 1023-y_raw[0]]
    return  x, y, x_raw[1] - x_raw[0], y_raw[1] - y_raw[0]

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
                time_lag = [3, 7], 
                color_line = 'k', 
                color_ref = 'r'):
    pol_r = flh.process_selection(pol_r)
    tor_rp = flh.process_selection(tor_r)
    sel = flh.process_selection(selection)

    x, y, dx, dy = return_view_xy(data)

    t = return_coord(data, 'Time')
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
            xp, yp = pixel_2_array(lines[0, i, tor_ind], lines[1, i, tor_ind], x[0], y[0], dx, dy)
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
            max_ccf = np.max(data_ccf.data, axis=2)
            max_p = np.argmax(data_ccf.data, axis=2)
            mid_p = int(np.floor(data_ccf.data.shape[2]) / 2)
            plot_corr(fig, 
                      axes[0,0], 
                      (max_p.T[::-1,:]-mid_p) * dt, 
                      -time_lag[0]*dt, time_lag[0]*dt, 
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
            plot_ref_line(axes[0,0], lines, i, tor_rp, color_ref)
            plot_corr(fig, 
                      axes[0,1], 
                      (max_p.T[::-1,:]-mid_p) * dt, 
                      -time_lag[1]*dt, time_lag[1]*dt, 
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
            plot_ref_line(axes[0,1], lines, i, tor_rp, color_ref)
            plot_corr(fig, 
                      axes[1,0], 
                      max_ccf.T[::-1,:], 
                      -corr_v, 
                      corr_v, 
                      'XCorr Max', 
                      lines[0, i, tor_ind], 
                      lines[1, i, tor_ind], 
                      lines[:, pol_r, :], 
                      tor_r, 
                      color_line,  
                      color_ref, 
                      title + ', Max CCF', 
                      surfs, 
                      x, 
                      y)
            plot_ref_line(axes[1,0], lines, i, tor_rp, color_ref)

            for j in range(3):
                plot_corr(fig, 
                        axes.ravel()[j + 3], 
                        data_ccf.data[:,:,mid_p-4+j].T[::-1,:], 
                        -corr_v, 
                        corr_v, 
                        'XCorr', 
                        lines[0, i, tor_ind], 
                        lines[1, i, tor_ind], 
                        lines[:, pol_r, :], 
                        tor_r, 
                        color_line,  
                        color_ref, 
                        title + f', CCF, t = {-44 + j*11}' +  r'$\mu$s', 
                        surfs, 
                        x, 
                        y)
                plot_ref_line(axes.ravel()[j + 3], lines, i, tor_rp, color_ref)
            pdf.savefig()

            fig, axes = plt.subplots(3,2, sharex=True, sharey=True)
            fig.set_size_inches(8.25, 11.75)
            axes[0,0].set_ylim(y[0], y[-1])
            axes[0,0].set_xlim(x[0], x[-1])

            for j in range(6):
                plot_corr(fig, 
                        axes.ravel()[j], 
                        data_ccf.data[:,:,mid_p-1+j].T[::-1,:], 
                        -corr_v, 
                        corr_v, 
                        'XCorr', 
                        lines[0, i, tor_ind], 
                        lines[1, i, tor_ind], 
                        lines[:, pol_r, :], 
                        tor_r, 
                        color_line,  
                        color_ref, 
                        title + f', CCF, t = {-11 + j*11}' +  r'$\mu$s', 
                        surfs, 
                        x, 
                        y)
                plot_ref_line(axes.ravel()[j], lines, i, tor_rp, color_ref)
            pdf.savefig()


def create_slides(data, 
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
    pol_r = flh.process_selection(pol_r)
    tor_rp = flh.process_selection(tor_r)
    sel = flh.process_selection(selection)

    x, y, dx, dy = return_view_xy(data)

    t = return_coord(data, 'Time')
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

def plot_ref_line(ax, lines, i, tor_rp, color_ref):
    ax.plot(lines[0, i, tor_rp], lines[1, i, tor_rp], c=color_ref, ls=':')
    ax.plot(lines[0, i, 0:260], lines[1, i, 0:260], c=color_ref, ls=':')
    ax.plot(lines[0, i, 7025:7302], lines[1, i, 7025:7302], c=color_ref, ls=':')

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
    roi_max = roi[0] + np.argmax(data_apsd.data[:,:,roi[0]:roi[1]], axis=2)
    for i in range(roi_max.shape[0]):
        for j in range(roi_max.shape[1]):
            roi_max[i, j] = f[roi_max[i, j]]
    return mean_spect, roi_power, roi_max

def binning_data(data):
    y_raw = return_coord(data, 'Image y')
    x_raw = return_coord(data, 'Image x')
    im_x = (x_raw[-1] - x_raw[0] + 1) // 3
    im_y = (y_raw[-1] - y_raw[0] + 1) // 3
    data_bin = np.zeros((im_x, im_y, data.shape[2]))
    for i in range(im_x):
        for j in range(im_y):
            data_bin[i,j,:] = np.mean(data.data[3*i:3*i+2,3*j:3*j+2,:], axis=(0,1))
    
    t = data.get_coordinate_object('Time')
    x = data.get_coordinate_object('Image x')
    y = data.get_coordinate_object('Image y')

    x = flap.Coordinate(name='Image x', 
                        mode=flap.CoordinateMode(equidistant=True), 
                        unit = x.unit, 
                        shape=x.shape, 
                        start=x.start, 
                        step=3, 
                        dimension_list=x.dimension_list)
    
    y = flap.Coordinate(name='Image y', 
                        mode=flap.CoordinateMode(equidistant=True), 
                        unit = y.unit, 
                        shape=y.shape, 
                        start=y.start, 
                        step=3, 
                        dimension_list=y.dimension_list)

    data_bin = flap.DataObject(data_array = data_bin, 
                               data_unit = data.data_unit, 
                               coordinates = [t, x, y], 
                               data_title = data.data_title + ', binned', 
                               exp_id = data.exp_id,
                               data_source = data.data_source,
                               info = data.info)
    return data_bin

def closest_fl_point(x, y, lines):
    p = np.array([[[x]], [[y]]])
    lines = lines - p
    lines = np.sqrt(lines[0, :, :]**2 + lines[1, :, :]**2)
    pol = np.argmin(np.min(lines, axis=1))
    tor = np.argmin(np.min(lines, axis=0))
    return pol, tor

def return_distance(x1, y1, x2, y2, lines, lines_2):
    pol1, tor1 = closest_fl_point(x1, y1, lines_2)
    pol2, tor2 = closest_fl_point(x2, y2, lines_2)
    return np.sqrt(np.sum((lines[:, pol1, tor1] - lines[:, pol2, tor2])**2))

def return_raw_data(shot, time):
    opts = {'Datapath' : '/data/W7-X', 'Time' : time, 'Timing path' : '/data/W7-X/processed_data/Photron/integ_data', 'Max_size' : 24}
    data = flap_w7x_camera.w7x_camera_get_data(shot, 'AEQ21_PHOTRON_ROIP1', options=opts)
    return data

def initial_process(shot, time, t0, tend, save_path, f_low=2000, f_high=11000):
    data = return_raw_data(shot, time)
    t = return_coord(data, 'Time')
    data = data.slice_data(slicing={'Time' : t[t0:tend]})
    data.detrend(coordinate='Time', options={'Trend removal': 'Mean'})
    shot = shot.replace('.', '_')
    data.get_coordinate_object('Time').mode.equidistant = True
    data.get_coordinate_object('Time').start = t[t0]
    data.get_coordinate_object('Time').step = t[t0 + 1] - t[t0]
    data.data = data.data.swapaxes(0,1)[::-1,:,:]
    data.get_coordinatet_object('Image x').start = x[0]
    data.get_coordinatet_object('Image y').start = y[0]
    data.save(save_path + '/' + shot + f'_{t0}.dat', protocol=4)
    filter_options = {'Type': 'Bandpass', 
                      'f_low': f_low, 
                      'f_high': f_high, 
                      'Tau': 1.1111111234640703e-05, 
                      'Design': 'Chebyshev II', 
                      'Steepness': 0.2, 
                      'Loss': 1, 
                      'Attenuation': 20}
    data_fil = data.filter_data(coordinate='Time', options=filter_options)
    data_fil.save(save_path + '/' + shot + f'_{t0}_fil_{f_low}Hz.dat', protocol=4)
    data = binning_data(data)
    data.save(save_path + '/' + shot + f'_{t0}_bin.dat', protocol=4)
    data_fil = data.filter_data(coordinate='Time', options=filter_options)
    data_fil.save(save_path + '/' + shot + f'_{t0}_bin_fil_{f_low}Hz.dat', protocol=4)
