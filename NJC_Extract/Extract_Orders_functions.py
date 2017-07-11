#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/17 at 2:09 PM

Last Modified on July 11 2017 19:26

@author: Neil Cook

Version 0.0.11
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.cm as cmaps
import os
import sys
from tqdm import tqdm
from skimage import measure
from skimage import filters

if sys.version_info[0] < 3:
    from collections import OrderedDict as dict
else:
    pass


# =============================================================================
# Define VARIABLES
# =============================================================================
# location of config file
configfile = '../config.txt'
# expected formats required for parameters
VARIABLES = dict(filepath=str, plotpath=str, savepath=str, files=list,
                 order_direction=str, xblur=float, yblur=float,
                 fitmin=int, fitmax=int, width=int, minpixelsinorder=int)
# descriptions for required parameters
DESCS = dict()
DESCS['filepath'] = ("string, location of the folder containing fits files" +
                    " to extract")
DESCS['plotpath'] = "string, location of place to save plots"
DESCS['savepath'] = "string, location to save extracted order fits files"
DESCS['poly_fits_file'] = ('string, file to save polynomials fits file to' +
                           ' will save in: (in $save path$/$input file$/ )')

DESCS['files'] = "list or string, names of image fits files to extract"
DESCS['order_direction'] = ("string, direction of the orders " +
                            "If 'horizontal' or 'H' or 'h' then orders are" +
                            "assumed to increase along the x axis" +
                            "If 'vertical' or 'V' or 'v' then orders are" +
                            "assumed to increase along the y axis")
DESCS['xblur'] = ("float, blur pixels in wavelength direction " +
                  "(to aid finding orders)")
DESCS['yblur'] = ("float, blur pixels in order direction " +
                  "(to aid finding orders)")
DESCS['pcut'] = ("float, percentile to remove background pixels at greats a " +
                 "mask of pixels to fit and not to fit")
DESCS['fitmin'] = "int, lowest order polynomial to fit"
DESCS['fitmax'] = "int, highest order polynomial to fit"
DESCS['width'] = "int, number of pixels to use for width of each order"
DESCS['minpixelsinorder'] = ("minimum number of pixels to use order" +
                             " (else it is discarded as noise/unusable)")
PORDER = ['filepath', 'plotpath', 'savepath', 'files', 'order_direction',
          'xblur', 'yblur', 'pcut', 'fitmin', 'fitmax', 'width',
          'minpixelsinorder']

# =============================================================================
# Define CMD line args and input/output
# =============================================================================
def cmdlineargs(p):
    """
    finds command lines inputs

    :return params:  dictionary, updated parameters to use in the program
    """
    # get args from command line (using sys.argv looking for '='
    cmdargs = sys.argv
    ckwargs = dict()
    for arg in cmdargs:
        if '=' in arg:
            key, value = arg.replace(' ', '').split('=')
            if key in VARIABLES:
                # try to convert to required type (using VARIABLES dict)
                try:
                    if VARIABLES[key] in [list, np.ndarray]:
                        ckwargs[key] = VARIABLES[key](value.split(','))
                    else:
                        ckwargs[key] = VARIABLES[key](value)
                except ValueError:
                    emsg = [key, str(VARIABLES[key])]
                    print('Command line input not understood for' +
                          'argument {0} must be {1}'.format(*emsg))
    # we need default values if not set by command line
    for key in p:
        if key not in ckwargs:
            ckwargs[key] = p[key]
    # return parameter dictionary
    return ckwargs


def textfileargs(p, cloc):
    """
    Imports parameters from text file located at cloc (by default this is
    set at the top of the program - if not changed below)

    :param p: dictionary, default parameter dictionary (the default values)
    :param cloc: string or None, location of config file

    :return:
    """

    emsg = 'Error in config file: '
    # load text file (remove all white spaces)
    data = np.genfromtxt(cloc, dtype=str, comments='#', delimiter='=')
    keys, values = data[:, 0], data[:, 1]
    for k in range(len(keys)):
        keys[k] = keys[k].replace(' ', '')
        values[k] = values[k].replace(' ', '')
    textdict = dict(zip(keys, values))
    # add root to paths
    paths = ['filepath', 'plotpath', 'savepath', 'polypath']
    for path in paths:
        p[path] = textdict['root'] + textdict[path]
        p[path] = p[path].replace('//', '/')
    # deal with poly fits files
    svar = 'poly_fits_file'
    if '.fit' not in textdict[svar]:
        emsg2 = '"{0}" (value of "{1}")'.format(svar, textdict[svar])
        raise Exception(emsg + emsg2 + ' must contain ".fit"')
    else:
        p[svar] = textdict[svar]
    # deal with file lists
    svar = 'files'
    p[svar] = textdict[svar].split(',')
    for lvar in p[svar]:
        if '.fit' not in lvar:
            emsg2 = '"{0}" (value of "{1}")'.format(svar, lvar)
            raise Exception(emsg + emsg2 + ' must contain ".fit"')
    # deal with order direction
    svar = 'order_direction'
    sargs = ['h', 'horizontal','v', 'vertical']
    sargs2 = 'H, h, Horizontal, V, v, or Vertical'
    if textdict[svar].lower() not in sargs:
        emsg2 = '"{0}" (value of "{1}")'.format(svar, textdict[svar])
        raise Exception(emsg + emsg2 + ' must contain: ' + sargs2)
    else:
        p[svar] = textdict[svar]
    # deal with floats
    svars = ['xblur', 'yblur', 'pcut']
    for svar in svars:
        emsg2 = '"{0}" (value of "{1}")'.format(svar, textdict[svar])
        try:
            p[svar] = float(textdict[svar])
        except ValueError:
            raise Exception(emsg + emsg2 + ' must be a float')
    # deal with integers
    svars = ['fitmin', 'fitmax','width', 'minpixelsinorder']
    for svar in svars:
        emsg2 = '"{0}" (value of "{1}")'.format(svar, textdict[svar])
        try:
            p[svar] = int(textdict[svar])
        except ValueError:
            raise Exception(emsg + emsg2 + ' must be an integer')
    # finally return parameters
    return p


def set_inputs(p):
    """
    Sets the inputs using user input

    :param p: dictionary, parameters to use in the program

    :return p: dictionary, updated parameters to use in the program
    """
    # loop through the default parameters
    print('\n\n' + '=' * 50)
    print('\tSetup inputs\n')
    for key in PORDER:
        if key not in p:
            continue
        cond2 = True
        while cond2:
            print('\n\n' + '='*50)
            description = ('\n' + DESCS[key] + '\n\n' + 'current value ' +
                           ' =\t{0}'.format(p[key]))
            q1args = [key, description]
            q1 = "\n\nModify {0}?\n\n\t{1}\n\n\tModify {0}? [Y]es or [N]o:\t"
            uinput0 = input(q1.format(*q1args))
            if 'Y' in uinput0.upper():
                uinput01 = input("\n\nEnter value for {0}\t:")
                try:
                    if VARIABLES[key] in [list, np.ndarray]:
                        p[key] = VARIABLES[key](uinput01.split(','))
                    else:
                        p[key] = VARIABLES[key](uinput01)
                    cond2 = False
                except ValueError:
                    emsg = [key, str(VARIABLES[key])]
                    print('Command line input not understood for' +
                          'argument {0} must be {1}'.format(*emsg))
                    cond2 = True
            elif 'N' in uinput0.upper():
                cond2 = False
            else:
                cond2 = True
    print('\n' + '=' * 50 + '\n\n')
    return p


def loaddata(filename, p):
    # make folder
    name = filename.split('.fit')[0]
    ppath = p['plotpath'] + '/' + name + '/'
    if name not in os.listdir(p['plotpath']):
        os.makedirs(ppath)
    fpath = p['savepath'] + '/' + name + '/'
    if name not in os.listdir(p['savepath']):
        os.makedirs(fpath)
    # --------------------------------------------------------------------
    # Load data
    ff = p['filepath'] + filename
    print('\n\t Loading CCD image for:\n\t {0}...'.format(ff))
    im = np.array(fits.getdata(ff))
    # --------------------------------------------------------------------
    # flip the image if orders are in the y direction
    # (default is x direction)
    if 'v' in p['order_direction'].lower():
        im = im.T
    # return the image and update params
    p['name'] = name
    p['ppath'] = ppath
    p['fpath'] = fpath
    return im, p


def save_fits(pfits, params, xls, xhs, spath):
    """
    save the fits to file, for use in extracting future spectra

    :param pfits: list, length same as number of orders, polynomial values
                      (output of np.polyval)

                      i.e. p where:

                      p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    """
    data = []
    cols = ['Order'] + list(range(params['fitmax'] + 1))[::-1]
    cols += ['low_x', 'high_x']

    # Loop around each order and plot fits
    for pp, pf in enumerate(pfits):
        row = [pp + 1] + list(np.zeros(params['fitmax'] + 1 - len(pf)))
        row += list(pf)
        row += [xls[pp], xhs[pp]]
        data.append(row)
    data = np.array(data)

    # convert to astropy table
    atable = Table()
    for c, col in enumerate(cols):
        atable[str(col)] = data[:, c]
    atable.write(spath + params['poly_fits_file'], overwrite=True)


def read_fits(filename, params):
    """
    read in the polynomial fit fits file (saved from Extract_Orders.py)
    and extract using polynomial fits

    :param filename: string, location and file name of the file containing the
                     polynomial fits (params['poly_fits_file'] by default)

    :param params: dict, parmaeter dictionary, must include the following
                         keywords:
                              - fitmax: int, highest order polynomial to fit
                                             the order to (default = 3)

    fits file should look like the following:

    Order          6                  5          ...       0        low_x   high_x
    float64      float64            float64       ...    float64    float64 float64
    ------- ------------------ ------------------ ... ------------- ------- -------
        1.0  7.80380819018e-20 -7.28455089523e-16 ... 1086.73647399     0.0  3028.0
        2.0  2.09138850722e-19 -2.01802656266e-15 ... 1123.38227429     0.0  3082.0
        3.0  1.56641316229e-19 -1.79748717147e-15 ... 1159.81599355     0.0  3203.0
        4.0  2.26957361856e-19 -2.31431373935e-15 ... 1196.07630084     0.0  3319.0

    where 6, 5, 4, 3, 2, 1, 0 are the polynomial powers in p

        i.e. p where:

            p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    :return:
    """
    # convert to astropy table
    atable = Table.read(filename)
    orders = np.array(atable['Order'], dtype=int)

    pfits1, xls, xhs = [], [], []
    for order in orders:
        pfit1 = []
        for p in range(params['fitmax']+1)[::-1]:
            pfit1.append(atable[str(p)][order-1])
        pfits1.append(pfit1)
        xls.append(int(atable['low_x'][order-1]))
        xhs.append(int(atable['high_x'][order-1]))

    return pfits1, xls, xhs


# =============================================================================
# Define Calculation functions
# =============================================================================
def min_max_clip(image, lowsigma, highsigma):
    """
    Performs a min max clip of the image

    :param image: numpy 2D array, containing the image to search

    :param lowsigma: float, multiplier of the std to take off the median
                     median - lowsigma * std becomes the minimum value to keep,
                     everything below gets set to this value

    :param highsigma: float, multiplier of the std to add to the median
                      median + highsigma * std maximum value to keep
                      everything below gets set to this value

    i.e. all values below (median - lowsigma * std) are set to
         (median - lowsigma * std)

         and

         all values above (median + highsigma * std) are set to
         (median + highsigma * std)

         returns an image
    """
    stats = ("\n Stats:")
    stats += ("\n   Median={0:.3f}, ".format(np.median(image)))
    stats += ("  1$\sigma$={0:.3f}".format(np.std(image)))
    # set all pixels above median + highsigma * sigma to 3 sigma
    image1 = np.array(image)
    lowv = np.median(image1) - lowsigma*np.std(image1)
    highv = np.median(image1) + highsigma*np.std(image1)
    highsig = image1 > highv
    lowsig = image1 < lowv
    image1[highsig] = highv
    image1[lowsig] = 0.0

    stats += ("\nLow is at {0}, ".format(lowv))
    stats += ("  High is at {0}".format(highv))
    # return altered image
    return image1, stats


def locate_orders(image, xb=4.0, yb=128.0, cutp=50.0, minpix=10000,
                  log=False):
    """
    Locate the orders using a blob finding algorithm

    :param image: numpy 2D array, containing the image to search

    :param xb: int, blur in the x direction (wavelength direction) use this
               to help identify orders by bluring out spectral information
               along the wavelength direction (default 128)

    :param yb: int, blur in the y direction (across orders), usually small,
               just used to remove any features across the order and make it
               easier to find the orders

    :param cutp: float, the percentile to cut at
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

                i.e. for case above

                       [0, 1, 2, 3]
    """
    # smooth out the data along the wavelength direction
    # (to remove faint regions and aid fitting along orders)
    if log:
        print("\n\t\tApplying blur...")
    if xb == 0 and yb == 0:
        image2 = np.array(image)
    else:
        image2 = filters.gaussian(image, sigma=[xb, yb])
    # mask out noise
    blobs = image2 > np.percentile(image2, cutp)

    # use skimage measure to differentitate orders
    if log:
        print("\n\t\tFinding order blobs...")
    dblobs = measure.label(blobs)
    # identify different regions
    rs = np.unique(dblobs)
    # require a minimum number of pixels in an order blob
    if log:
        print("\n\t\tRemoving blobs with under {0} pixels...".format(minpix))
    for r1 in list(rs):
        rmask = dblobs == r1
        if len(dblobs[rmask]) < minpix:
            dblobs[rmask] = 0.0
    # identify different regions (after cull)
    rs = np.unique(dblobs)
    # return orders (dblobs) and region list (rs)
    if log:
        print("\n\t\tOrder finding complete.")
    return dblobs, rs


def fit_orders(rs, dblobs, fmin=1, fmax=3):
    """
    Fits a a set of polynomails (of order fmin to fmax) to each order chooses
    lowest chisquared value as polynomial fit for that order

    :param rs: numpy 1D array, the unique integers found in dblobs

                i.e. for case above

                       [0, 1, 2, 3]

    :param dblobs: numpy 2D array equal in shape to the input image, contains
                    located "blobs" which should be the orders (if they were
                    detectable) format is 0 where no order is found and then
                    a different integer for each "blob" found

                    i.e.:

                        010203
                        010203
                        010203
                        010203

                        identifies three orders

    :param fmin: int, lowest order polynomial to fit the order to (default = 1)

    :param fmax: int, highest order polynomial to fit the order to (default = 3)

    :return pfits: list, length same as number of orders, polynomial values
                      (output of np.polyval)

                      i.e. p where:

                      p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    :return forders: list, length same as number of orders, contains choosen
                       polynomial order for each order (int)

    :return xls: list, length same as number of orders, the lowest x pixel
                   (wavelength direction) used in each order

    :return xhs: list, length same as number of orders, the highest x pixel
                   (wavelength direction) used in each order

    """
    pfits, forders, xls, xhs = [], [], [], []
    # loop around each order (blob regions)
    for region in rs:
        # ignore the 0 region (background region)
        if region == 0:
            continue
        # create a mask for this region
        mask = dblobs == region
        # Extract x and y positions of this mask
        px, py = np.where(mask)
        # save the min and max fit points
        xls.append(np.min(px))
        xhs.append(np.max(px))
        # fit a polynomial to this region
        pfs, ns = polynomialfit(px, py, nmin=fmin, nmax=fmax)
        pfits.append(pfs), forders.append(ns)
    return pfits, forders, xls, xhs


def remove_orders(polyfits, rm_orders, xlows, xhighs):
    if type(rm_orders) is not list:
        return polyfits, xlows, xhighs
    mask = np.repeat([True], len(polyfits))
    # Remove orders (if rm_orders is populated)
    for r in range(len(polyfits)):
        if r + 1 in rm_orders:
            mask[r] = False
    return mask


def polynomialfit(x, y, nmin=1, nmax=3):
    """
    Select the minimum chi squared polynomial fit between nmin and nmax

    :param x: x axis array
    :param y: y axis array
    :param nmin: minimum order to fit
    :param nmax: maximum order to fit

    Note could improve this to weight it by pixel uncertainties
    """
    chis, ps = [], []
    nrange = range(nmin, nmax+1)
    for n in nrange:
        p = np.polyfit(x, y, n)
        ps.append(p)
        chis.append(np.sum((y-np.polyval(p, x))**2))
    argmin = np.argmin(chis)
    return ps[argmin], nrange[argmin]


def extract_orders(image, pfits, xls, xhs, w=10):
    """
    Extract the spectra from each of the orders and return in a list of numpy
    arrays

    :param image: numpy 2D array, containing the CCD image

    :param pfits: list, length same as number of orders, polynomial values
                      (output of np.polyval)

                      i.e. p where:

                      p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
    :param xls: list, length same as number of orders, the lowest x pixel
                   (wavelength direction) used in each order

    :param xhs: list, length same as number of orders, the highest x pixel
                   (wavelength direction) used in each order

    :param w: int, number of pixels either side of the polynomial to sum
              (extract) to create the spectrum pixels

    :return spec: list,  length same as number of orders, each containing
                  a 1D numpy array with the spectrum of each order in

    """
    spec = []
    xarr = range(0, len(image))
    for pp, pf in enumerate(pfits):
        print('\n\t\t Order {0}'.format(pp + 1))
        # find nearest y pixel
        xarr1 = range(xls[pp], xhs[pp])
        yarr = np.array(np.round(np.polyval(pf, xarr), 0), dtype=int)
        # loop around x pixels
        ospec = []
        for xrow in tqdm(xarr):
            # find the lower bound for extraction of y (i.e. need to deal with
            # going out of bounds
            if yarr[xrow] - w < 0:
                lower = 0
            else:
                lower = yarr[xrow] - w
            # find the upper bound for extraction of y
            if yarr[xrow] + w > len(image[0]):
                upper = len(image[0])
            else:
                upper = yarr[xrow] + w
            # extract and sum pixels to form this x pixel (wavelength) count
            if xrow in xarr1:
                ssum = np.sum(image[xrow, lower: upper])
                ospec.append(ssum)
            else:
                ospec.append(np.NaN)
        spec.append(ospec)
    return spec


def save_orders(spec, spath):
    """
    Save each order to file
    :param spec: list,  length same as number of orders, each containing
                  a 1D numpy array with the spectrum of each order in
    :param spath: path to save the fits file to
                  (columns = pixel number and counts)
    """
    for s in range(len(spec)):
        nt = Table()
        nt['pixel number'] = np.arange(0, len(spec[s]))
        nt['counts'] = spec[s]
        nt.write(spath + 'Order_{0}.fits'.format(s), overwrite=True)


def run_extraction(image, pfits, xls, xhs, params, name, ppath, fpath):
    # Extract fitted orders
    print('\n\t Extracting Orders...')
    spectra = extract_orders(image, pfits, xls, xhs, params['width'])
    # --------------------------------------------------------------------
    # Save fitted orders
    print('\n\t Saving Orders...')
    save_orders(spectra, fpath)
    # --------------------------------------------------------------------
    # Plot fitted orders
    print('\n\t Plotting Ordered Spectra...')
    plot_orders(pfits, spectra, name, ppath)


# =============================================================================
# Define plotting functions
# =============================================================================
def plot_image(image, name=None, spath=None, show=False, return_frame=False,
               frame=None):
    """
    Plots the raw CCD image to $spath$/$n$_Original[.png,.pdf]

    :param image: numpy 2D array, containing the CCD image

    :param n: string, name of the CCD image (extracted from file name) forms
              prefix of plot filename

    :param spath: path to save the plot to

    :param show: if True then plot is shown instead of saved

    :return:
    """
    if frame is None:
        fig, frame = plt.subplots(1, 1)
        fig.set_size_inches(32, 32)
    frame.imshow(image, cmap='gray')
    frame.set_xticklabels([])
    frame.set_yticklabels([])
    frame.xaxis.set_ticks_position('none')
    frame.yaxis.set_ticks_position('none')
    if return_frame:
        return frame
    elif show:
        print("\n\n Please close the graph (Figure 1) to continue (Alt F4).\n\n")
        plt.show()
    else:
        sname = spath + name + '_Original'
        plt.savefig(sname + '.png', bbox_inches='tight')
        # plt.savefig(sname + '.pdf', bbox_inches='tight')
        plt.close()


def plot_found_orders(dblobs, n=None, spath=None, show=False, return_frame=True,
                      frame=None):
    """
    Plots the found orders over the raw CCD image to
    $spath$/$n$_FoundOrders[.png,.pdf]

    :param dblobs: numpy 2D array equal in shape to the input image, contains
                    located "blobs" which should be the orders (if they were
                    detectable) format is 0 where no order is found and then
                    a different integer for each "blob" found

                    i.e.:

                        010203
                        010203
                        010203
                        010203

                        identifies three orders

    :param n: string, name of the CCD image (extracted from file name) forms
              prefix of plot filename

    :param spath: path to save the plot to

    :param show: if True then plot is shown instead of saved

    :return:
    """
    if frame is None:
        fig, frame = plt.subplots(1, 1)
        fig.set_size_inches(32, 32)
    frame.patch.set_facecolor('black')
    # frame.imshow(image, cmap='gray')
    # ddd = np.ma.masked_where(dblobs == 0, dblobs)
    uorders = np.unique(dblobs)
    ublobs = np.array(dblobs)
    for it in range(len(uorders)):
        oit = uorders[it]
        if oit == 0:
            continue
        mask = ublobs == oit
        if np.sum(mask) == 0:
            continue
        ublobs[mask] = it

    cmap = cmaps.jet
    cmap.set_bad('black', 0.0)
    frame.imshow(ublobs, cmap='jet')
    frame.set_xticklabels([])
    frame.set_yticklabels([])
    frame.xaxis.set_ticks_position('none')
    frame.yaxis.set_ticks_position('none')
    sname = str(spath) + str(n) + '_FoundOrders'


    if return_frame:
        return frame
    elif show:
        print("\n\n Please close the graph (Figure 1) to continue (Alt F4).\n\n")
        plt.show()
    else:
        plt.savefig(sname + '.png', bbox_inches='tight')
        # plt.savefig(sname + '.pdf', bbox_inches='tight')
        plt.close()


def plot_fitted_orders(image, pfits, xls, xhs, n=None, spath=None, show=False,
                       return_frame=True, frame=None, mask=None):
    """
    Plots the fitted orders over the raw CCD image to
    $spath$/$n$_FittedOrders[.png,.pdf]

    :param image: numpy 2D array, containing the CCD image

    :param pfits: list, length same as number of orders, polynomial values
                      (output of np.polyval)

                      i.e. p where:

                      p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    :param xls: list, length same as number of orders, the lowest x pixel
                   (wavelength direction) used in each order

    :param xhs: list, length same as number of orders, the highest x pixel
                   (wavelength direction) used in each order

    :param n: string, name of the CCD image (extracted from file name) forms
              prefix of plot filename

    :param spath: path to save the plot to


    :param show: if True then plot is shown instead of saved

    :return:
    """
    if mask is None:
        print('mask none')
        mask = np.repeat([True], len(pfits))
    if frame is None:
        fig, frame = plt.subplots(1, 1)
        fig.set_size_inches(32, 32)
    frame.imshow(image, cmap='gray')

    names = np.arange(1, len(pfits)+1)
    # Loop around each order and plot fits
    for pp, pf in enumerate(pfits):
        if mask[pp] == False:
            continue
        else:
            xarr = np.arange(xls[pp], xhs[pp], 1)
            yarr = np.polyval(pf, xarr)
            yarr[yarr < 0] = 0
            yarr[yarr > len(yarr)] = len(yarr)

            ymid = np.median(yarr)
            xmid = np.median(xarr)

            frame.plot(yarr, xarr, color='r', linewidth=2)
            frame.text(ymid, xmid, 'Order{0}'.format(names[pp]),
                       horizontalalignment='center', verticalalignment='center',
                       rotation=90, color='g', zorder=5)

    # finalise graph
    frame.set_xlim([0, len(image[0])*1.1])
    frame.set_ylim([0, len(image)])
    frame.set_xticklabels([])
    frame.set_yticklabels([])

    if return_frame:
        return frame
    elif show:
        print("\n\n Please close the graph (Figure 1) to continue (Alt F4).\n\n")
        plt.show()
    else:
        sname = spath + n + '_FittedOrders'
        plt.savefig(sname + '.png', bbox_inches='tight')
        # plt.savefig(sname + '.pdf', bbox_inches='tight')
        plt.close()


def plot_orders(pfits, spec, n, spath):
    """
    Plots the extracted order spectra counts vs pixel location
    $spath$/$n$_Spectrum_order_{0}[.png,.pdf]

    :param pfits: list, length same as number of orders, polynomial values
                      (output of np.polyval)

                      i.e. p where:

                      p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    :param spec: list,  length same as number of orders, each containing
                  a 1D numpy array with the spectrum of each order in

    :param n: string, name of the CCD image (extracted from file name) forms
              prefix of plot filename

    :param spath: path to save the plot to

    :return:
    """
    for pp, pf in enumerate(pfits):
        fig, frame = plt.subplots(1, 1)
        fig.set_size_inches(10, 10)
        frame.plot(spec[pp])
        frame.set_xlabel('Pixel value')
        frame.set_ylabel('Counts')
        frame.set_title('Order no. {0}'.format(pp + 1))
        sname = spath + n + '_Spectrum_order_{0}'.format(pp + 1)
        plt.savefig(sname + '.png', bbox_inches='tight')
        # plt.savefig(sname + '.pdf', bbox_inches='tight')
        plt.close()