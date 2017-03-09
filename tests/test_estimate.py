#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from radial import estimate, dataset

# Import data from file
w16 = dataset.RVDataSet('HIP67620_WF16.dat', t_offset=-5E4,
                        rv_offset='subtract_mean', instrument_name='W16',
                        target_name='HIP 67620', t_col=1, rv_col=3,
                        rv_unc_col=4)

# Setup first guess and search bounds
guess = {'log_k': np.log10(6314),
         'log_period': np.log10(3819.2),
         't0': 4904.5,
         'omega': 139.3 * np.pi / 180,
         'log_ecc': np.log10(0.343),
         'sqe_cosw': np.sqrt(0.343) * np.cos(139.3 * np.pi / 180),
         'sqe_sinw': np.sqrt(0.343) * np.sin(139.3 * np.pi / 180),
         'gamma_0': 127,
         'sigma_0': 154}

bounds = {'log_k': (3, 4),
          'log_period': (3.0, 4.0),
          't0': (4000, 6000),
          'omega': (120 * np.pi / 180, 160 * np.pi / 180),
          'log_ecc': (-0.5, -0.4),
          'gamma_0': (0, 1000),
          'sigma_0': (0, 1000)}

# Perform LMFIT estimation using the MC10 parametrization
estim = estimate.FullOrbit([w16], guess, bounds, use_add_sigma=False,
                           parametrization='mc10')
result_mc10 = estim.lmfit_orbit(update_guess=False, verbose=False)

# Perform LMFIT estimation using the EXOFAST parametrization
estim = estimate.FullOrbit([w16], guess, bounds, use_add_sigma=False,
                           parametrization='exofast')
result_exofast = estim.lmfit_orbit(update_guess=False, verbose=False)

# Test emcee estimation
result_emcee = estim.emcee_orbit(nthreads=2)