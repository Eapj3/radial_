#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from radial import object
import astropy.units as u


A = object.MainStar(mass=0.954 * u.solMass)
B = object.Companion(k=6.314 * u.km / u.s,
                     period_orb=3819.2 * u.d,
                     t_0=4904.5 * u.d,
                     omega=139.3 * u.deg,
                     ecc=0.343)

HIP67620 = object.System(main_star=A, companions=[B], name='HIP67620')
HIP67620.mass_func()
print(HIP67620.companions[0].msini)