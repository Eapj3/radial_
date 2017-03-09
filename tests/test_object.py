#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from radial import object
import astropy.units as u
import matplotlib.pyplot as plt


A = object.MainStar(mass=0.954 * u.solMass)
B = object.Companion(k=6.314 * u.km / u.s,
                     period_orb=3819.2 * u.d,
                     t_0=4904.5 * u.d,
                     omega=139.3 * u.deg,
                     ecc=0.343)
b = object.Companion(k=0.5 * u.km / u.s,
                     period_orb=700 * u.d,
                     t_0=2000.5 * u.d,
                     omega=180.3 * u.deg,
                     ecc=0.543)

# time = (np.loadtxt('HIP67620_WF16.dat', usecols=(1,)) - 5E4) * u.d
time = np.linspace(2000, 7000, 1000) * u.d
HIP67620 = object.System(main_star=A, companion=[B, b], name='HIP67620',
                         time=time)
HIP67620.mass_func()

HIP67620.compute_rv()
HIP67620.plot_rv()
plt.show()