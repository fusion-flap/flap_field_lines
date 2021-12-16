# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 2021

@author: lordofbejgli
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    fig, ax = plt.figure()
    plt.xlim(xlim)
    plt.ylim(ylim)
    return fig, ax

def stamp_surfs(tor_r, view, color='y', direction='backward'):
    _, s2 = get_surfs('10:95:10', tor_r, direction, view)
    for i in range(9):
        plt.plot(s2[0, :, i], s2[1, :, i], color)

def stamp_lines(lines_2, tor_r, color='r'):
    tor_r = flh.process_selection(tor_r)
    for i in range(lines_2.shape[1]):
        plt.plot(lines_2[0, i, tor_r], lines_2[1, i, tor_r], color)

def stamp_surfs_2(surfs_2, color='y'):
    for i in range(surfs_2.shape[2]):
        plt.plot(surfs_2[0, :, i], surfs_2[1, :, i], color)

def plot_frame(data, frame, x=[316, 707], y=[384, 639]):
    fig = give_plot(x,y)
    im = plt.imshow(data.data[:,:,frame].T[::-1,:], extent=[x[0], x[-1], y[0], y[-1]])
    return im, fig

def pixel_2_array(x, y, x0, y0):
    x = x - x0
    y = y-y0
    return np.rint(x), np.rint(y)

def array_2_pixel(x, y, x0, y0):
    x = x + x0
    y = y + y0
    return x, y

def plot_corr(fig, ax, data, vmin, vmax, label, xref, yref, lines, range, color, title, x=[316, 707], y=[384, 639]):
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    im = ax.imshow(data, cmap='bwr', vmin=vmin, vmax=vmax, extent=[x[0], x[-1], y[0], y[-1]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    stamp_lines(lines, range, color)
    ax.plot(xref, yref, 'c*')
    ax.set_title(title)

def create_book(data, savefile, pol_r, tor_r, lines, surfs, ind):
    pol_r = flh.process_selection(pol_r)
    tor_rp = flh.process_selection(tor_r)

    y_raw = data.coordinate('Image y', options={'Change only' : True})[0]
    y_raw = np.squeeze(y_raw)
    x_raw = data.coordinate('Image x', options={'Change only' : True})[0]
    x_raw = np.squeeze(x_raw)
    y = np.linspace(x_raw[0], 1023-x_raw[0], 256)
    x = np.linspace(y_raw[0], 1023-y_raw[0], 392)

    t = np.squeeze(data.coordinate('Time', options={'Change only' : True})[0])
    dt = (t[1] - t[0]) * 10**6
    t  = flap.Coordinate(name='Time', 
                         mode=flap.CoordinateMode(equidistant=True), 
                         unit = 'Second', 
                         shape=(22000), 
                         start=t[0], 
                         step=t[1] - t[0], 
                         dimension_list=[0])

    dunit  = flap.Unit(name='Amplitude',unit='V')

    with PdfPages(savefile) as pdf:
	    for i in pol_r:
		    xp, yp = pixel_2_array(surfs[0, i, ind], surfs[1, i, ind], x[0], y[0])
		    fig, axes = plt.subplot(3,2, sharex=True, sharey=True)
		    fig.set_size_inches(8.25, 11.75)
            axes[0,0].set_ylim(y[0], y[-1])
            axes[0,0].set_xlim(x[0], x[-1])
            d_ref = data.data[xp, yp, :]
            d_ref = flap.DataObject(data_array = d_ref, data_unit = dunit, coordinates = [t], data_title = 'Reference', exp_id = '123456_01')
            data_ccf = data.ccf(ref=d_ref, coordinate='Time', options={'Normalize' : True})
            max_p = np.argmax(data_ccf.data, axis=2)
            plot_corr(fig, axes[0,0], (max_p.T[::-1,:]-275) * dt, -5*dt, 5*dt, r'Time lag ($\mu$s)', surfs[0, i, ind], surfs[1, i, ind], lines, tor_r, 'w:')
            plt.plot(lines[0, ind, tor_rp], lines[1, ind, tor_rp], 'k:')
            pdf.savefig()