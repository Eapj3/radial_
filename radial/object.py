#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
import astropy.constants as c

"""
This module contains the massive object classes that make up a star-planet
system.
"""


# The main star class
class MainStar(object):
    """

    """
    def __init__(self, mass, name=None):
        self.mass = mass
        self.name = name


# The companion sub-class
class Companion(object):
    """

    Parameters
    ----------
    k : ``astropy.units.Quantity`` or ``None``, optional

    period_orb : ``astropy.units.Quantity`` or ``None``, optional

    t_0 : ``astropy.units.Quantity`` or ``None``, optional

    omega : ``astropy.units.Quantity`` or ``None``, optional

    ecc : ``float`` or ``None``, optional

    msini : ``astropy.units.Quantity`` or ``None``, optional

    semi_a : ``astropy.units.Quantity`` or ``None``, optional

    name : ``str`` or ``None``, optional
    """
    def __init__(self, k=None, period_orb=None, t_0=None, omega=None, ecc=None,
                 msini=None, semi_a=None, name=None, main_star=None):
        self.main_star = main_star
        self.msini = msini
        self.semi_a = semi_a
        self.k = k
        self.period_orb = period_orb
        self.t_0 = t_0
        self.omega = omega
        self.ecc = ecc
        self.name = name


# The star-companion system class
class System(object):
    """

    Parameters
    ----------
    main_star : ``radial.object.MainStar``
        The main star of the system.

    companions : list
        Python list containing the all the ``radial.object.Companion`` of the
        system.

    name : ``str`` or ``None``, optional
        Name of the system. Default is ``None``.
    """
    def __init__(self, main_star, companions, name=None):
        self.name = name
        self.main_star = main_star
        self.companions = companions
        self.n_c = len(self.companions)     # Number of companions

        # Initializing useful global parameters
        self.f = []

    # Compute mass functions of the companions
    def mass_func(self):
        """

        Returns
        -------

        """
        for comp in self.companions:
            assert isinstance(comp.k, u.Quantity), 'k needs to be provided.'
            assert isinstance(comp.period_orb, u.Quantity), 'period_orb ' \
                                                            'needs to be ' \
                                                            'provided.'
            assert isinstance(comp.ecc, float), 'ecc needs to be provided.'
            f = (comp.period_orb * comp.k ** 3 * (1 - comp.ecc ** 2) ** (3 / 2)
                 / (2 * np.pi * c.G)).to(u.solMass)
            m_star = self.main_star.mass
            m_roots = np.roots([1,  -f.value,  -2 * m_star.value * f.value,
                                -m_star.value ** 2 * f.value])
            comp.msini = abs(m_roots[0]) * u.solMass
            comp.semi_a = (np.sqrt(c.G / comp.k * comp.msini * comp.period_orb /
                           (2 * np.pi * np.sqrt(1 - comp.ecc ** 2)))).to(u.AU)
            self.f.append(f)