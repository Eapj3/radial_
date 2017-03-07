#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from radial import orbit

t_sim = np.linspace(3000, 5000, 1000)

# First, we create an instance of the system HIP156846
HIP156846 = orbit.BinarySystem(k=464,
                               period=359.51,
                               t0=3998.1,
                               omega=52.2 * np.pi / 180,
                               ecc=0.847,
                               gamma=0.0)

# The RVs are computed simply by running get_rvs()
rvs_mc10 = HIP156846.get_rvs(ts=t_sim)

# Now, using the 'exofast' parametrization
HIP156846 = orbit.BinarySystem(k=464,
                               period=359.51,
                               t0=3998.1,
                               sqe_cosw=np.sqrt(0.847) * np.cos(52.2 *
                                                                np.pi / 180),
                               sqe_sinw=np.sqrt(0.847) * np.sin(52.2 *
                                                                np.pi / 180),
                               gamma=0.0)
rvs_exofast = HIP156846.get_rvs(ts=t_sim)

# Check if the radial velocities from both parametrizations are consistent
assert rvs_mc10[:10].all != rvs_exofast[:10].all
