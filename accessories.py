# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 2021

@author: lordofbejgli
"""

import numpy as np
import matplotlib.pyplot as plt
import flap_field_lines.field_line_handler as flh

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
    return fig

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

def plot_corr(data, vmin, vmax, label, xref, yref, lines, range, color, x=[316, 707], y=[384, 639]):
    give_plot(x, y)
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    im = plt.imshow(data, cmap='bwr', vmin=vmin, vmax=vmax, extent=[x[0], x[-1], y[0], y[-1]])
    cbar = plt.colorbar()
    cbar.set_label(label)
    stamp_lines(lines, range, color)
    plt.plot(xref, yref, 'c*')
