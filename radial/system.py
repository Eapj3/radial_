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


# The catch-all massive object class
class MassiveObject(object):
    """

    """
    def __init__(self, mass, name=None):
        self.name = name
        self.mass = mass


# The main star sub-class
class MainStar(MassiveObject):
    """

    """
    def __init__(self, mass, name=None, companions=None):
        super(MainStar, self).__init__(mass, name)
        self.companions = companions


# The companion sub-class
class Companion(MassiveObject):
    """

    Parameters
    ----------
    k : ``astropy.units.Quantity`` or ``None``, optional

    period_orb : ``astropy.units.Quantity`` or ``None``, optional

    t_0 : ``astropy.units.Quantity`` or ``None``, optional

    omega : ``astropy.units.Quantity`` or ``None``, optional

    ecc : ``astropy.units.Quantity`` or ``None``, optional

    msini : ``astropy.units.Quantity`` or ``None``, optional

    semi_a : ``astropy.units.Quantity`` or ``None``, optional

    mass : ``astropy.units.Quantity`` or ``None``, optional

    name : ``str`` or ``None``, optional

    main_star : ``radial.system.MainStar`` or ``None``, optional

    """
    def __init__(self, k=None, period_orb=None, t_0=None, omega=None, ecc=None,
                 msini=None, semi_a=None, mass=None, name=None, main_star=None):

        super(Companion, self).__init__(mass, name)
        self.main_star = main_star
        self.msini = msini
        self.semi_a = semi_a
        self.k = k
        self.period_orb = period_orb
        self.t_0 = t_0
        self.omega = omega
        self.ecc = ecc

        # Initializing useful global parameters
        self.f = None

    # Compute mass function
    def mass_func(self):
        """

        Returns
        -------

        """
        assert isinstance(self.k, u.Quantity), 'k needs to be provided.'
        assert isinstance(self.period_orb, u.Quantity), 'period_orb needs to ' \
                                                        'be provided.'
        assert isinstance(self.ecc, u.Quantity), 'ecc needs to be provided.'
        f = (self.period_orb * self.k ** 3 * (1 - self.ecc ** 2) ** (3 / 2) /
             (2 * np.pi * c.G)).to(u.solMass)
        m_star = self.main_star.mass
        m_roots = np.roots([1,  -f.value,  -2 * m_star.value * f.value,
                            -m_star.value ** 2 * f.value])
        self.msini = abs(m_roots) * u.solMass
        self.semi_a = (np.sqrt(c.G / self.k * self.msini * self.period_orb /
                             (2 * np.pi * np.sqrt(1 - self.ecc ** 2)))).to(u.AU)
        self.f = f
        return f
