#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from radial import estimate, dataset
import matplotlib.pyplot as plt

# Import data from file
w16 = dataset.RVDataSet('HIP67620_WF16.dat', t_offset=-5E4,
                        rv_offset='subtract_mean', instrument_name='W16',
                        target_name='HIP 67620', t_col=1, rv_col=3,
                        rv_unc_col=4)

# Setup first guess and search bounds
guess = {'k': 6314,
         'period': 3819.2,
         't0': 4904.5,
         'omega': 139.3 * np.pi / 180,
         'ecc': 0.343,
         'gamma_0': 127,
         'sigma_0': 50}

# Perform LMFIT estimation using the MC10 parametrization
estim = estimate.FullOrbit([w16], guess, use_add_sigma=True,
                           parametrization='mc10')
result_mc10 = estim.lmfit_orbit(update_guess=True, verbose=True)

# Perform LMFIT estimation using the EXOFAST parametrization
#estim = estimate.FullOrbit([w16], guess, use_add_sigma=True,
#                           parametrization='exofast')
#result_exofast = estim.lmfit_orbit(update_guess=True, verbose=True)

# Test emcee estimation
#result_emcee = estim.emcee_orbit(nthreads=2)