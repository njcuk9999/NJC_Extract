#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:56:52 2016

Last Modified on July 11 2017 19:26

@author: Neil Cook

Program to extract Echelle spectra orders from a CCD image
(From scratch)

Version 0.0.11
"""

import sys
import numpy as np
import NJC_Extract as ne

# detect python version
# if python 3 do this:
if (sys.version_info > (3, 0)):
    pass
# if python 2 do this:
else:
    def input(message):
        return raw_input(message)


# =============================================================================
# Define variables
# =============================================================================
DPARAMS = dict()     # default param dictionary
# set up paths
WORKSPACE = '/local/home/ncook/Projects/Side-Project-Hugh/Stage2/'
WORKSPACE = '/Astro/Projects/Hughs_Work/'
# location of config file
CONFIGFILE = './config.txt'
# path where the data are located
DPARAMS['root'] = '/'
DPARAMS['filepath'] = WORKSPACE + '/Data/'
# path to save the plots to
DPARAMS['plotpath'] = WORKSPACE + '/Plots/NeilPipeline/'
# path to save the fits files to
DPARAMS['savepath'] = WORKSPACE + '/Data/NeilPipeline/'
# file to save polynomial fits to (in save path/file/ )
DPARAMS['poly_fits_file'] = 'All_saved_polynomials.fits'
# files as a list of strings
DPARAMS['files'] = ['1154.fit']   # must be a list
# -------------------------------------------------------------------------
DPARAMS['order_direction'] = 'H' # If "horizontal" or "H" or "h" then orders are
                                 # assumed to increase along the x axis
                                 # If "vertical" or "V" or "v" then orders are
                                 # assumed to increase along the y axis
DPARAMS['xblur'] = 64    # blur pixels in wavelength direction
                          # (to aid finding orders)
DPARAMS['yblur'] = 10   # blur pixels in order direction
                          # (to aid finding orders)
DPARAMS['pcut'] = 50      # percentile to remove background pixels at
                          # greats a mask of pixels to fit and not to fit
DPARAMS['fitmin'] = 1     # lowest order polynomial to fit
DPARAMS['fitmax'] = 6     # highest order polynomial to fit
DPARAMS['width'] = 10     # number of pixels to use for width of each order
DPARAMS['minpixelsinorder'] = 10000  # minimum number of pixels to use order
                                     # (else it is discarded as noise/unusable)

GUI = False

# =============================================================================
# Start of code
# =============================================================================
# deal with arguments from a text file
params = ne.textfileargs(DPARAMS, CONFIGFILE)
# deal with arguments from command line
params = ne.cmdlineargs(params)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # loop around params['files']
    for filename in params['files']:
        # ---------------------------------------------------------------------
        while True:
            uinput = input("\nSet up input parameters? [Y]es [N]o:\t")
            if 'Y' in uinput.upper():
                params = ne.set_inputs(params)
            else:
                break
        # ---------------------------------------------------------------------
        cond, im1, diffblobs = True, np.array([]), np.array([])
        while cond:
            # -----------------------------------------------------------------
            # Print progress
            ne.cmdtitle('Analysing file: {0}'.format(filename))
            # -----------------------------------------------------------------
            # load the CCD image
            im, params = ne.loaddata(filename, params)
            # -----------------------------------------------------------------
            # Fitting the orders (min max clip + locating + fitting)
            cond = True
            im1, diffblobs = np.array([]), np.array([])
            # -----------------------------------------------------------------
            # min max clipping routine
            ne.cmdtitle('Initial clip of image')
            im1 = ne.run_min_max_clip(im, userinput=GUI)
            # -----------------------------------------------------------------
            # locating orders
            ne.cmdtitle('Locating Orders')
            args = [im1, params['xblur'], params['yblur'], params['pcut'],
                    params['minpixelsinorder']]
            diffblobs, regions = ne.run_locating_orders(*args, userinput=GUI)
            # -----------------------------------------------------------------
            # Fit each order with a polynomial fit
            print('\n\t Fitting orders with polynomial fits')
            args = [regions, diffblobs, params['fitmin'], params['fitmax']]
            polyfits, fitorders, xlows, xhighs = ne.fit_orders(*args)
            # -----------------------------------------------------------------
            # Reorder polynomials by max y value (highest to lowest)
            ymaxs = []
            xarr0 = range(0, len(im1))
            for polyfit in polyfits:
                ymaxs.append(np.max(np.polyval(polyfit, xarr0)))
            sortmask = np.argsort(ymaxs)
            polyfits = np.array(polyfits)[sortmask]
            # -----------------------------------------------------------------
            # run remove orders and check whether we accept fits
            # cond breaks loop if fits accepted
            print('\n\t Allowing user to remove orders...')
            cond = ne.run_remove_orders(im1, polyfits, xlows, xhighs, regions,
                                        userinput=GUI)
        # ---------------------------------------------------------------------
        ne.cmdtitle('Fitting complete')
        # ---------------------------------------------------------------------
        # Plot image
        print('\n\t Plotting Image...')
        ne.plot_image(im, params['name'], params['ppath'])
        # Plot found orders overlay image
        print('\n\t Plotting Found Orders...')
        ne.plot_found_orders(diffblobs, params['name'], params['ppath'])
        # ---------------------------------------------------------------------
        # Plot fitted orders
        print('\n\t Plotting Fitted Orders...')
        ne.plot_fitted_orders(im, polyfits, xlows, xhighs, params['name'],
                              params['ppath'])
        # ---------------------------------------------------------------------
        # saving polyinomal fits to file
        print('\n\t Saving polynomial fits to file...')
        ne.save_fits(polyfits, params, xlows, xhighs, params['fpath'])
        # ---------------------------------------------------------------------
        # extraction code
        ne.run_extraction(im, polyfits, xlows, xhighs, params, params['name'],
                          params['ppath'], params['fpath'])



# =============================================================================
# End of code
# =============================================================================
