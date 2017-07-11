#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:56:52 2016

Last Modified on July 6 2017 12:24

@author: Neil Cook

Program to extract Echelle spectra orders from a CCD image
(using a pre-generated polynomial fit file)

Version 0.0.11
"""

import os
import sys

import numpy as np
from astropy.io import fits

import NJC_Extract as ne

# detect python version
# if python 3 do this:
if (sys.version_info > (3, 0)):
    raise SystemError("Error, only supported in Python 2 only")
# if python 2 do this:
else:
    def input(message):
        print(message)
        raw = sys.stdin.readline()
        return raw.replace('\n', '')


# =============================================================================
# Define variables
# =============================================================================
dparams = dict()     # default param dictionary
# set up paths
workspace = '/local/home/ncook/Projects/Side-Project-Hugh/Stage2/'
# workspace = '/Astro/Projects/Hughs_Work/'
# location of config file
configfile = './config.txt'
# path where the data are located
dparams['filepath'] = workspace + '/Data/'
# path to save the plots to
dparams['plotpath'] = workspace + '/Plots/NeilPipeline/'
# path to save the fits files to
dparams['savepath'] = workspace + '/Data/NeilPipeline/'
# path to load polynomial fits from
dparams['polypath'] = workspace + '/Data/NeilPipeline/Savedfits/'
# file to save polynomial fits to (in save path/file/ )
dparams['poly_fits_file'] = 'All_saved_polynomials.fits'
# files as a list of strings
dparams['files'] = ['CCDImage1.fit']   # must be a list
# -------------------------------------------------------------------------
dparams['order_direction'] = 'H' # If "horizontal" or "H" or "h" then orders are
                                 # assumed to increase along the x axis
                                 # If "vertical" or "V" or "v" then orders are
                                 # assumed to increase along the y axis
dparams['xblur'] = 64    # blur pixels in wavelength direction
                          # (to aid finding orders)
dparams['yblur'] = 10   # blur pixels in order direction
                          # (to aid finding orders)
dparams['pcut'] = 50      # percentile to remove background pixels at
                          # greats a mask of pixels to fit and not to fit
dparams['fitmin'] = 1     # lowest order polynomial to fit
dparams['fitmax'] = 6     # highest order polynomial to fit
dparams['width'] = 10     # number of pixels to use for width of each order
dparams['minpixelsinorder'] = 10000  # minimum number of pixels to use order
                                     # (else it is discarded as noise/unusable)
# expected formats required for parameters
variables = dict(filepath=str, plotpath=str, savepath=str, files=list,
                 order_direction=str, xblur=float, yblur=float,
                 fitmin=int, fitmax=int, width=int, minpixelsinorder=int)
# descriptions for required parameters
descs = dict()
descs['filepath'] = ("string, location of the folder containing fits files" +
                    " to extract")
descs['plotpath'] = "string, location of place to save plots"
descs['savepath'] = "string, location to save extracted order fits files"
descs['poly_fits_file'] = ('string, file to save polynomials fits file to' +
                           ' will save in: (in $save path$/$input file$/ )')

descs['files'] = "list or string, names of image fits files to extract"
descs['order_direction'] = ("string, direction of the orders " +
                            "If 'horizontal' or 'H' or 'h' then orders are" +
                            "assumed to increase along the x axis" +
                            "If 'vertical' or 'V' or 'v' then orders are" +
                            "assumed to increase along the y axis")
descs['xblur'] = ("float, blur pixels in wavelength direction " +
                  "(to aid finding orders)")
descs['yblur'] = ("float, blur pixels in order direction " +
                  "(to aid finding orders)")
descs['pcut'] = ("float, percentile to remove background pixels at greats a " +
                 "mask of pixels to fit and not to fit")
descs['fitmin'] = "int, lowest order polynomial to fit"
descs['fitmax'] = "int, highest order polynomial to fit"
descs['width'] = "int, number of pixels to use for width of each order"
descs['minpixelsinorder'] = ("minimum number of pixels to use order" +
                             " (else it is discarded as noise/unusable)")
porder = ['filepath', 'plotpath', 'savepath', 'files', 'order_direction',
          'xblur', 'yblur', 'pcut', 'fitmin', 'fitmax', 'width',
          'minpixelsinorder']



# =============================================================================
# Start of code
# =============================================================================
# deal with arguments from command line
params = ne.cmdlineargs(dparams)
# deal with arguments from a text file
params = ne.textfileargs(params, configfile)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # loop around params['files']
    for f in params['files']:
        # --------------------------------------------------------------------
        while True:
            uinput = input("\nSet up input parameters? [Y]es [N]o:\t")
            if 'Y' in uinput.upper():
                params = ne.set_inputs(params)
            else:
                break
        # --------------------------------------------------------------------
        # Print progress
        print('\n\n\n\n\n\n{0}\n\n'.format('='*50))
        print('\n Analysing file: {0}'.format(f))
        print('\n\n{0}\n\n'.format('='*50))
        name = f.split('.fit')[0]
        ppath = params['plotpath'] + '/' + name + '/'
        if name not in os.listdir(params['plotpath']):
            os.makedirs(ppath)
        fpath = params['savepath'] + '/' + name + '/'
        if name not in os.listdir(params['savepath']):
            os.makedirs(fpath)
        # --------------------------------------------------------------------
        # Load data
        ff = params['filepath'] + f
        print('\n\n\t Loading CCD image for:\n\t {0}...'.format(ff))
        im = np.array(fits.getdata(ff))
        # --------------------------------------------------------------------
        # flip the image if orders are in the y direction
        # (default is x direction)
        if 'v' in params['order_direction'].lower():
            im = im.T
        # read poly fits from file
        readargs = [params['polypath'] + params['poly_fits_file'], params]
        polyfits, xlows, xhighs = ne.read_fits(*readargs)
        # --------------------------------------------------------------------
        # Plot fitted orders
        print('\n\t Plotting Fitted Orders...')
        ne.plot_fitted_orders(im, polyfits, xlows, xhighs, name, ppath)
        # --------------------------------------------------------------------
        # extraction code
        ne.run_extraction(im, polyfits, xlows, xhighs, params, name, ppath, fpath)



# =============================================================================
# End of code
# =============================================================================
