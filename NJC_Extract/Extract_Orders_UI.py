#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/17 at 12:56 PM

Last Modified on July 11 2017 19:26

@author: Neil Cook

Version 0.0.11
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from . import Extract_Orders_functions as EOF
from . import tkcanvas as tkc

# detect python version
if sys.version_info[0] < 3:
    import Tkinter as Tk
    from collections import OrderedDict as dict
else:
    import tkinter as Tk


# =============================================================================
# UI functions
# =============================================================================
def run_min_max_clip(image, lowclip=0.0, highclip=2.0, userinput=True):
    if userinput:
        # set up plot
        fig, frame = plt.subplots(1, 1)
        # get initial image
        image1, statstr = EOF.min_max_clip(image, lowclip, highclip)
        # define keywords for
        pkwargs = dict(frame=frame, image=image1, lowclip=lowclip,
                       highclip=highclip)

        # define update plot function
        def plot(frame, image, lowclip, highclip):
            frame.clear()
            title = 'Min max clipping \n lower sigma={0} high sigma={1}'
            title = title.format(lowclip, highclip)
            image, statstr = EOF.min_max_clip(image, lowclip, highclip)
            title += statstr
            frame = EOF.plot_image(image, return_frame=True, frame=frame)
            frame.set_title(title)

        # run initial update plot function
        plot(**pkwargs)

        # define widgets
        widgets = dict()
        widgets['lowclip'] = dict(label='Enter low sigma clip',
                                  comment='Enter a new value for low sigma '
                                          'value\n sigma=0 will clip at median'
                                          ' \n sigma<0 will do median + sigma'
                                          ' \n sigma>0 will do median - sigma',

                                  kind='TextEntry', minval=-100.0, maxval=100.0,
                                  fmt=float, start=lowclip)
        widgets['highclip'] = dict(label='Enter high sigma clip',
                                   comment='Enter a new value for high sigma '
                                           'value\n sigma=0 will clip at median'
                                           ' \n sigma<0 will do median - sigma'
                                           ' \n sigma>0 will do median + sigma',

                                   kind='TextEntry', minval=-100.0,
                                   maxval=100.0,
                                   fmt=float, start=highclip)
        widgets['close'] = dict(label='Next', kind='ExitButton',
                                position=Tk.BOTTOM)
        widgets['update'] = dict(label='Update', kind='UpdatePlot',
                                 position=Tk.BOTTOM)

        wprops = dict(orientation='v', position=Tk.RIGHT)

        gui1 = tkc.TkCanvas(figure=fig, ax=frame, func=plot, kwargs=pkwargs,
                            title='Min max clipping', widgets=widgets,
                            widgetprops=wprops)
        gui1.master.mainloop()

        # assign new low and high clip
        if 'lowclip' in gui1.data:
            if gui1.data['lowclip'] is not None:
                lowclip = gui1.data['lowclip']
        if 'highclip' in gui1.data:
            if gui1.data['highclip'] is not None:
                highclip = gui1.data['highclip']

    image1, statstr = EOF.min_max_clip(image, lowclip, highclip)
    return image1



def run_locating_orders(image, hblur, vblur, pcut, minpix, userinput=True):
    """
    User interface locating the orders  repeats until user is satisfied with
    the locating

    :param image: numpy 2D array, containing the CCD image

    :param hblur: int, blur in the x direction (wavelength direction) use this
                  to help identify orders by bluring out spectral information
                  along the wavelength direction (default 128)

    :param vblur: int, blur in the y direction (across orders), usually small,
                  just used to remove any features across the order and make it
                  easier to find the orders

    :param pcut: float, the percentile to cut at
                  i.e.
                    50.0 = median (50%)
                    15.9 = 1 sigma below (16%)
                    84.1 = 1 sigma above (84%)

    :param minpix: int, minimum number of pixels needed to deem order useful
                   (any orders with less than this number of pixels are
                    ignored) - useful to remove blobs not associated with orders
                    or orders with too few pixels to be usable

    :return dblobs: numpy 2D array equal in shape to the input image, contains
                    located "blobs" which should be the orders (if they were
                    detectable) format is 0 where no order is found and then
                    a different integer for each "blob" found

                    i.e.:

                        010203
                        010203
                        010203
                        010203

                        identifies three orders

    :return rs: numpy 1D array, the unique integers found in dblobs
    """
    if not userinput:
        return EOF.locate_orders(image, hblur, vblur, pcut, minpix)
    # set up plot
    fig, frame = plt.subplots(1, 1)
    # get initial blobs and regions
    dblobs, rs = EOF.locate_orders(image, hblur, vblur, pcut, minpix)
    # define keywords for
    pkwargs = dict(frame=frame, image=image, dblobs=dblobs, hblur=hblur,
                   vblur=vblur, pcut=pcut, minpix=minpix)

    # define update plot function
    def plot(frame, image, dblobs, hblur, vblur, pcut, minpix, log=False):
        frame.clear()
        title = 'Found orders'
        title += ('\n  horizontal blur = {0} [pixels]'.format(hblur))
        title += ('\n  vertical blur = {0} [pixels]'.format(vblur))
        title += ('\n  cut percentile = {0} [%]'.format(pcut))
        title += ('\n  min pixels selected = {0} [pixels]'.format(minpix))
        dblobs, rs = EOF.locate_orders(image, hblur, vblur, pcut, minpix,
                                       log=log)
        frame = EOF.plot_found_orders(dblobs, return_frame=True, frame=frame)
        frame.set_title(title)

    # run initial update plot function
    plot(log=True, **pkwargs)

    # define widgets
    widgets = dict()
    widgets['hblur'] = dict(label='Enter amount of x blurring',
                            comment='blur in the x direction (wavelength \n'
                                    'direction) use this to help identify \n'
                                    'orders by bluring out spectral \n'
                                    'information along the wavelength \n'
                                    'direction',
                            kind='TextEntry', minval=0.001, maxval=None,
                            fmt=float, start=float(hblur))
    widgets['vblur'] = dict(label='Enter amount of y blurring',
                            comment='blur in the y direction (across orders),\n'
                                    ' usually small, just used to remove any \n'
                                    'features across the order and make it \n'
                                    'easier to find the orders',
                            kind='TextEntry', minval=0.001, maxval=None,
                            fmt=float, start=float(vblur))
    widgets['pcut'] = dict(label='Enter percentile to cut at',
                           comment='i.e. \n'
                                   '  50.0 = median (50%)\n'
                                   '  15.9 = 1 sigma below (16%)\n'
                                   '  84.1 = 1 sigma above (84%)\n',
                           kind='TextEntry', minval=0.0, maxval=100.0,
                           fmt=float, start=pcut)
    widgets['minpix'] = dict(label='Enter minimum number of pixels',
                             comment='minimum number of pixels needed to \n'
                                     'deem order useful (any orders with \n'
                                     'less than this number of pixels are \n'
                                     'ignored) - useful to remove blobs \n'
                                     'not associated with orders or orders \n'
                                     'with too few pixels to be usable',
                             kind='TextEntry', minval=1, maxval=None,
                             fmt=int, start=int(minpix))

    widgets['close'] = dict(label='Next', kind='ExitButton',
                            position=Tk.BOTTOM)
    widgets['update'] = dict(label='Update', kind='UpdatePlot',
                             position=Tk.BOTTOM)

    wprops = dict(orientation='v', position=Tk.RIGHT)

    gui2 = tkc.TkCanvas(figure=fig, ax=frame, func=plot, kwargs=pkwargs,
                        title='Locating orders', widgets=widgets,
                        widgetprops=wprops)
    gui2.master.mainloop()

    # assign new low and high clip
    if 'hblur' in gui2.data:
        if gui2.data['hblur'] is not None:
            hblur = gui2.data['hblur']
    if 'vblur' in gui2.data:
        if gui2.data['vblur'] is not None:
            vblur = gui2.data['vblur']
    if 'pcut' in gui2.data:
        if gui2.data['pcut'] is not None:
            pcut = gui2.data['pcut']
    if 'minpix' in gui2.data:
        if gui2.data['minpix'] is not None:
            minpix = gui2.data['minpix']

    dblobs, rs = EOF.locate_orders(image, hblur, vblur, pcut, minpix)
    return dblobs, rs


def run_remove_orders(im1, polyfits, xlows, xhighs, regions, userinput=True):
    # Plot fitted orders
    if not userinput:
        return polyfits, xlows, xhighs, True

    # convert to numpy arrays
    polyfits = np.array(polyfits)
    xlows, xhighs = np.array(xlows), np.array(xhighs)

    # set up plot
    fig, frame = plt.subplots(1, 1)

    rm_orders = []
    # get kwargs
    pkwargs = dict(frame=frame, im1=im1, polyfits=polyfits, xlows=xlows,
                   xhighs=xhighs, rm_orders=rm_orders)

    # define update plot function
    def plot(frame, im1, polyfits, xlows, xhighs, rm_orders):
        frame.clear()
        title = ('Removing bad orders  \n(Largest order '
                 'number = {0})'.format(len(polyfits)))
        mask = EOF.remove_orders(polyfits, rm_orders, xlows, xhighs)
        EOF.plot_fitted_orders(im1, polyfits, xlows, xhighs, mask=mask,
                               return_frame=True, frame=frame)
        frame.set_title(title)

    # run initial update plot function
    plot(**pkwargs)

    # define valid_function
    # input is one variable (the string input)
    # return is either:
    #   True and values
    # or
    #   False and error message
    def vfunc(xs):
        try:
            xwhole = xs.replace(',', ' ')
            new_xs = xwhole.split()
            xs = []
            for nxs in new_xs:
                xs.append(int(nxs))
            return True, xs
        except:
            return False, ('Error, input must consist of integers \n '
                           'separated by commas or white spaces')

    # define widgets
    widgets = dict()
    widgets['rm_orders'] = dict(label='Select orders to remove',
                                comment='Enter all order numbers to remove \n'
                                        'separated by a whitespace or comma \n'
                                        'to undo just delete the entered '
                                        'number',
                                kind='TextEntry', minval=None, maxval=None,
                                fmt=str, start=" ", valid_function=vfunc,
                                width=60)

    widgets['accept'] = dict(label='Accept Orders', kind='ExitButton',
                             position=Tk.BOTTOM)
    widgets['start'] = dict(label='Start fitting \n process again',
                            kind='ExitButton', result='False', onclick='True',
                            position=Tk.BOTTOM)
    widgets['update'] = dict(label='Update', kind='UpdatePlot',
                             position=Tk.BOTTOM)

    wprops = dict(orientation='v', position=Tk.RIGHT)

    gui3 = tkc.TkCanvas(figure=fig, ax=frame, func=plot, kwargs=pkwargs,
                        title='Locating orders', widgets=widgets,
                        widgetprops=wprops)
    gui3.master.mainloop()

    if 'rm_orders' in gui3.data:
        if gui3.data['rm_orders'] is not None:
            if type(gui3.data['rm_orders']) == list:
                rm_orders = gui3.data['rm_orders']

    fmask = EOF.remove_orders(polyfits, rm_orders, xlows, xhighs)

    polyfits, xlows, xhighs = polyfits[fmask], xlows[fmask], xhighs[fmask]

    if 'start' in gui3.data:
        if gui3.data['start'] == 'True':
            print('\n\n\n\n\n\n')
            cmdtitle('Restarting fitting process', key='!')
            print('\n\n\n\n\n\n')
            return polyfits, xlows, xhighs, True
    else:
        return polyfits, xlows, xhighs, False


# =============================================================================
# cmd line functions
# =============================================================================
def cmdtitle(title, key='='):
    # print('\n\n\n\n\n\n{0}\n\n'.format('=' * 50))
    print('\n{0} {1} {0}\n'.format(key * 10, title))
    # print('\n\n{0}'.format('=' * 50))

# =============================================================================
# End of code
# =============================================================================
