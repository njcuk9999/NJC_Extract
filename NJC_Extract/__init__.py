#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/17 at 2:03 PM

@author: neil

Program description here

Version 0.0.11
"""

from . import Extract_Orders_functions
from . import Extract_Orders_UI
import os

__author__ = "Neil Cook"
__email__ = 'neil.james.cook@gmail.com'
__version__ = '0.0.11'
__location__ = os.path.realpath(os.path.join(os.getcwd(),
                                             os.path.dirname(__file__)))

# =============================================================================
# cmd/input functions
# =============================================================================
cmdlineargs = Extract_Orders_functions.cmdlineargs
textfileargs = Extract_Orders_functions.textfileargs
set_inputs = Extract_Orders_functions.set_inputs
loaddata = Extract_Orders_functions.loaddata
save_fits = Extract_Orders_functions.save_fits
read_fits = Extract_Orders_functions.read_fits

# =============================================================================
# calculation functions
# =============================================================================
min_max_clip = Extract_Orders_functions.min_max_clip
locate_orders = Extract_Orders_functions.locate_orders
fit_orders = Extract_Orders_functions.fit_orders
remove_orders = Extract_Orders_functions.remove_orders
polynomialfit = Extract_Orders_functions.polynomialfit
extract_orders = Extract_Orders_functions.extract_orders
save_orders = Extract_Orders_functions.save_orders
run_extraction = Extract_Orders_functions.run_extraction

# =============================================================================
# Plot functions
# =============================================================================
plot_image = Extract_Orders_functions.plot_image
plot_found_orders = Extract_Orders_functions.plot_found_orders
plot_fitted_orders = Extract_Orders_functions.plot_fitted_orders
plot_orders = Extract_Orders_functions.plot_orders

# =============================================================================
# UI functions
# =============================================================================
cmdtitle = Extract_Orders_UI.cmdtitle
run_min_max_clip = Extract_Orders_UI.run_min_max_clip
run_locating_orders = Extract_Orders_UI.run_locating_orders
run_remove_orders = Extract_Orders_UI.run_remove_orders