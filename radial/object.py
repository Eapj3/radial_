#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
from radial import orbit

"""
This module contains the massive object classes that make up a star-planet
system.
"""


# The main star class
class MainStar(object):
    """
    The main star of a given system.

    mass : ``astropy.units.Quantity``
        The mass of the star.

    name : ``str`` or ``None``
        The name of the star. Default is ``None``.
    """
    def __init__(self, mass, name=None):
        self.mass = mass
        self.name = name

        # The radial velocities corresponding to each individual companion
        self.radial_v_ind = None

        # The total radial velocities
        self.radial_v = None


# The companion sub-class
class Companion(object):
    """
    The massive companion class. It can be either a binary star, an exoplanet,
    maybe even a black hole! It can be anything that has a mass and orbits
    another massive object. General relativity effects are not implemented yet.

    Parameters
    ----------
    k : ``astropy.units.Quantity`` or ``None``, optional
        The radial velocity semi-amplitude. Default is ``None``.

    period_orb : ``astropy.units.Quantity`` or ``None``, optional
        The orbital period. Default is ``None``.

    t_0 : ``astropy.units.Quantity`` or ``None``, optional
        The time of periastron passage. Default is ``None``.

    omega : ``astropy.units.Quantity`` or ``None``, optional
        Argument of periapse. Default is ``None``.

    ecc : ``float`` or ``None``, optional
        Eccentricity of the orbit. Default is ``None``.

    msini : ``astropy.units.Quantity`` or ``None``, optional
        Mass of the companion multiplied by sine of the inclination of the
        orbital plane in relation to the line of sight. Default is ``None``.

    semi_a : ``astropy.units.Quantity`` or ``None``, optional
        Semi-major axis of the orbit. Default is ``None``.

    name : ``str`` or ``None``, optional
        Name of the companion. Default is ``None``.
    """
    def __init__(self, k=None, period_orb=None, t_0=None, omega=None, ecc=None,
                 msini=None, semi_a=None, name=None, main_star=None, mass=None,
                 sini=None):
        self.main_star = main_star
        self.msini = msini
        self.semi_a = semi_a
        self.k = k
        self.period_orb = period_orb
        self.t_0 = t_0
        self.omega = omega
        self.ecc = ecc
        self.name = name
        self.mass = mass
        self.sini = sini

        # Computing k from msini and a
        if self.sini is None:
            self.sini = 1
        else:
            pass
        if self.k is None:
            self.k = self.mass * 2 * np.pi * self.semi_a * self.sini / \
                (self.main_star.mass + self.mass) / self.period_orb / \
                np.sqrt(1 - self.ecc ** 2)
        else:
            pass


# The star-companion system class
class System(object):
    """
    The star-companions system class.

    Parameters
    ----------
    main_star : ``radial.object.MainStar``
        The main star of the system.

    companion : list
        Python list containing the all the ``radial.object.Companion`` of the
        system.

    time : ``astropy.units.Quantity`` or ``None``
        A scalar or ``numpy.ndarray`` containing the times in which the radial
        velocities are measured. Default is ``None``.

    name : ``str`` or ``None``, optional
        Name of the system. Default is ``None``.

    dataset : sequence, ``radial.dataset.RVDataSet`` or ``None``, optional
        A list of ``RVDataSet`` objects or one ``RVDataSet`` object that
        contains the data to be fit. Default is ``None``.
    """
    def __init__(self, main_star, companion, time=None, name=None,
                 dataset=None):
        self.name = name
        self.main_star = main_star
        self.companion = companion
        self.n_c = len(self.companion)     # Number of companions
        self.time = time
        self.dataset = dataset

        # Initializing useful global parameters
        self.f = []

    # Compute mass functions of the companions
    def mass_func(self):
        """
        Compute the mass functions of all the companions of the system. This
        method will also compute the msini and semi_a of the companions and
        save the values in their respective parameters.
        """
        for comp in self.companion:
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

    # Compute the radial velocities of the main star
    def compute_rv(self):
        """
        Compute the radial velocities of the main star, both the individual RVs
        (corresponding to each companion) and the total RVs.
        """
        ts = self.time.to(u.d).value
        star = self.main_star
        star.radial_v_ind = []
        for comp in self.companion:
            subsystem = orbit.BinarySystem(k=comp.k.to(u.m / u.s).value,
                                           period=comp.period_orb.to(u.d).value,
                                           t0=comp.t_0.to(u.d).value,
                                           omega=comp.omega.to(u.rad).value,
                                           ecc=comp.ecc)
            star.radial_v_ind.append(subsystem.get_rvs(ts))
        star.radial_v_ind = np.array(star.radial_v_ind)

        # Combine the individual radial velocities to compose the total
        star.radial_v = np.sum(star.radial_v_ind, axis=0) * u.m / u.s
        star.radial_v_ind = star.radial_v_ind * u.m / u.s

    # Plot radial velocities of the main star
    def plot_rv(self, companion_index=None, plot_title=None):
        """

        Parameters
        ----------
        companion_index : ``int`` or ``None``
            The companion index indicates which set of radial velocities will be
            plotted. If ``None``, then the total radial velocities are plotted.
            Default is ``None``.

        Returns
        -------
        fig :

        ax :
        """
        fig, ax = plt.subplots()
        star = self.main_star
        i = companion_index

        if i is None:
            ax.plot(self.time, star.radial_v)
            ax.set_xlabel('Time ({})'.format(str(self.time.unit)))
            ax.set_ylabel('Radial velocities ({})'.format(
                str(star.radial_v.unit)))
            ax.set_title(plot_title)
        else:
            ax.plot(self.time, star.radial_v_ind[i])
            ax.set_xlabel('Time ({})'.format(str(self.time.unit)))
            ax.set_ylabel('Radial velocities ({})'.format(
                str(star.radial_v_ind[i].unit)))
            ax.set_title(plot_title)

        return fig, ax
