#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
import scipy.optimize as sp

"""
This module computes the radial velocities of a massive object being orbited by
a secondary massive object, and is based on the formalism from Murray & Correia
(2011), available freely at http://arxiv.org/abs/1009.1738. The equation numbers
are from this article, unless otherwise noted.
"""


class BinarySystem(object):
    """
    A class that computes the radial velocities given the orbital parameters of
    the binary system.

    Parameters
    ----------
    k : scalar or ``astropy.units.Quantity``
        The radial velocity semi-amplitude K (velocity unit). If passed as
        scalar, assumes unit on ``work_units``.

    period : scalar or ``astropy.units.Quantity``
        The orbital period (time unit). If passed as scalar, assumes unit on
        ``work_units``.

    t0 : scalar or ``astropy.units.Quantity``
        Time of pariastron passage (time unit). If passed as scalar, assumes
        unit on ``work_units``.

    omega : scalar, ``astropy.units.Quantity`` or ``None``, optional
        Argument of periapse (angle unit). If ``None``, both ``sqe_cosw`` and
        ``sqe_sinw`` will be required. If passed as scalar, assumes unit on
        ``work_units``. Default is ``None``.

    ecc : scalar or ``None``, optional
        Eccentricity of the orbit. If ``None``, both ``sqe_cosw`` and
        ``sqe_sinw`` will be required. Default is ``None``.

    sqe_cosw : scalar or ``None``, optional
        The square root of the eccentricity multiplied by the cosine of the
        argument of periapse. If ``None``, both ``omega`` and ``ecc`` will be
        required. Default is ``None``.

    sqe_sinw : scalar or ``None``, optional
        The square root of the eccentricity multiplied by the sine of the
        argument of periapse. If ``None``, both ``omega`` and ``ecc`` will be
        required. Default is ``None``.

    gamma : scalar or ``astropy.units.Quantity``, optional
        Proper motion of the barycenter (velocity unit). If passed as scalar,
        assumes unit on ``work_units``. Default is 0.

    work_units : ``dict`` or ``None``, optional
        Dictionary containing the ``astropy.units`` for the entries 'velocity',
        'time' and 'angle'. If ``None``, assume units km / s, days and degrees,
        respectively.
    """
    def __init__(self, k, period, t0, omega=None, ecc=None, sqe_cosw=None,
                 sqe_sinw=None, gamma=0, work_units=None):

        if work_units is None:
            self.work_units = {'velocity': u.km / u.s, 'time': u.d, 'angle': u.deg}

        if omega is None or ecc is None:
            if sqe_cosw is None or sqe_sinw is None:
                raise ValueError('Either of these pairs have to be provided: '
                                 '(omega, ecc) or (sqe_cosw, sqe_sinw)')
            else:
                self.ecc = sqe_sinw ** 2 + sqe_cosw ** 2
                self.omega = np.arctan2(sqe_sinw, sqe_cosw)
        else:
            self.ecc = ecc
            if isinstance(omega, u.Quantity):
                self.omega = omega
            else:
                self.omega = omega * self.work_units['angle']

        if isinstance(k, u.Quantity):
            self.k = k
        else:
            self.k = k * self.work_units['velocity']
        if isinstance(period, u.Quantity):
            self.period = period
        else:
            self.period = period * self.work_units['time']
        if isinstance(t0, u.Quantity):
            self.t0 = t0
        else:
            self.t0 = t0 * self.work_units['time']
        if isinstance(gamma, u.Quantity):
            self.gamma = gamma
        else:
            self.gamma = gamma * self.work_units['velocity']

        if self.ecc > 1:
            raise ValueError('Keplerian orbits are ellipses, therefore ecc <= '
                             '1')

    # Compute Eq. 65
    def rv_eq(self, f):
        """
        The radial velocities equation.

        Parameters
        ----------
        f : ``astropy.unit.Quantity``
            True anomaly (unit of angle).

        Returns
        -------
        rvs : ``astropy.unit.Quantity``
            Radial velocity
        """
        rv = self.gamma + self.k * (np.cos(self.omega + f) + self.ecc *
                                    np.cos(self.omega))
        return rv

    # Calculates the Kepler equation (Eq. 41)
    def kep_eq(self, e_ano, m_ano):
        """
        The Kepler equation.

        Parameters
        ----------
        e_ano : ``astropy.unit.Quantity``
            Eccentric anomaly (unit of angle)

        m_ano : ``astropy.unit.Quantity``
            Mean anomaly (unit of angle)

        Returns
        -------
        kep: ``astropy.unit.Quantity``
            Value of E-e*sin(E)-M
        """
        kep = e_ano - self.ecc * np.sin(e_ano * u.rad) - m_ano
        return kep

    # Calculates the radial velocities for given orbital parameters
    def get_rvs(self, ts):
        """
        Computes the radial velocity given the orbital parameters.

        Parameters
        ----------
        ts : ``astropy.unit.Quantity``
            Time

        nt : int
            Number of points for one period. Default=1000.

        Returns
        -------
        rvs : ``astropy.unit.Quantity``
            Radial velocities
        """
        m_ano = 2 * np.pi / self.period * (ts - self.t0)  # Mean anomaly
        e_ano = np.array([sp.newton(func=self.kep_eq, x0=mk, args=(mk,))
                          for mk in m_ano]) * u.rad       # Eccentric anomaly
        # Computing the true anomaly
        f = 2 * np.arctan2(np.sqrt(1. + self.ecc) * np.sin(e_ano / 2),
                           np.sqrt(1. - self.ecc) * np.cos(e_ano / 2))
        # Why do we compute the true anomaly in this weird way? Because
        # arc-cosine is degenerate in the interval 0-360 degrees.
        rvs = self.rv_eq(f)
        return rvs


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    print('---------------------------------------')
    print('Starting test of keppy.orbit\n')
    t_sim = np.linspace(3000, 5000, 1000) * u.d
    start_time = time.time()  # We use this to measure the computation time

    # First, we create an instance of the system HIP156846
    HIP156846 = BinarySystem(k=0.464 * u.km / u.s,
                             period=359.51 * u.d,
                             t0=3998.1 * u.d,
                             #omega=52.2 * u.deg,
                             #ecc=0.847,
                             sqe_cosw=np.sqrt(0.847) * np.cos(52.2 * u.deg),
                             sqe_sinw=np.sqrt(0.847) * np.sin(52.2 * u.deg),
                             gamma=-68.54 * u.km / u.s)

    # The RVs are computed simply by running get_rvs()
    _rvs = HIP156846.get_rvs(ts=t_sim)
    print('RV calculation took %.4f seconds' % (time.time() - start_time))

    # Plotting results
    plt.plot(t_sim.value, _rvs.value)
    plt.xlabel('Time ({})'.format(t_sim.unit))
    plt.ylabel('RV ({})'.format(_rvs.unit))
    plt.show()
