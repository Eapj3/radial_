#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
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
    log_k : scalar
        Natural logarithm of the radial velocity semi-amplitude K [km/s]

    log_period : scalar
        Natural logarithm of the orbital period [days]

    t0 : scalar
        Time of pariastron passage [JD - 2.45E6 days]

    w : scalar or ``None``, optional
        Argument of periapse [degrees]. If ``None``, both ``sqe_cosw`` and
        ``sqe_sinw`` will be required. Default is ``None``.

    log_e : scalar or ``None``, optional
        Natural logarithm of the eccentricity. If ``None``, both ``sqe_cosw``
        and ``sqe_sinw`` will be required. Default is ``None``.

    sqe_cosw : scalar or ``None``, optional
        The square root of the eccentricity multiplied by the cosine of the
        argument of periapse. If ``None``, both ``w`` and ``log_e`` will be
        required. Default is ``None``.

    sqe_sinw : scalar or ``None``, optional
        The square root of the eccentricity multiplied by the sine of the
        argument of periapse. If ``None``, both ``w`` and ``log_e`` will be
        required. Default is ``None``.

    vz : scalar
        Proper motion of the barycenter [km/s]
    """
    def __init__(self, log_k, log_period, t0, w=None, log_e=None, sqe_cosw=None,
                 sqe_sinw=None, vz=0.0):

        if w is None or log_e is None:
            if sqe_cosw is None or sqe_sinw is None:
                raise ValueError('Either of these pairs have to be provided: '
                                 '(w, log_e) or (sqe_cosw, sqe_sinw)')
            else:
                self.e = sqe_sinw ** 2 + sqe_cosw ** 2
                self.w_rad = np.arctan2(sqe_sinw, sqe_cosw)
        else:
            self.e = 10 ** log_e
            self.w_rad = w * np.pi / 180.

        self.k = 10 ** log_k
        self.period = 10 ** log_period
        self.t0 = t0
        self.vz = vz

        if self.e > 1:
            raise ValueError('Keplerian orbits are ellipses, therefore e <= 1')

    # Compute Eq. 65
    def vr(self, f):
        """
        The radial velocities equation.

        Parameters
        ----------
        f : scalar or `numpy.ndarray`
            True anomaly [radians]

        Returns
        -------
        _rvs : scalar or array
            Radial velocities [km/s]
        """
        rvs = self.vz + self.k * (np.cos(self.w_rad + f) + self.e *
                                  np.cos(self.w_rad))
        return rvs

    # Calculates the Kepler equation (Eq. 41)
    def kep_eq(self, e_ano, m_ano):
        """
        The Kepler equation.

        Parameters
        ----------
        e_ano : scalar or array
            Eccentric anomaly [radians]

        m_ano : scalar or array
            Mean anomaly [radians]

        Returns
        -------
        kep: scalar or array
            Value of E-e*sin(E)-M
        """
        kep = e_ano - self.e * np.sin(e_ano) - m_ano
        return kep

    # Calculates the radial velocities for given orbital parameters
    def get_rvs(self, ts=np.linspace(0, 1, 1000), nt=1000, fold=False):
        """
        Computes the radial velocity given the orbital parameters.

        Parameters
        ----------
        ts : scalar or array
            Time [d]

        nt : int
            Number of points for one period. Default=1000.

        fold : bool, optional
            Switch to phase-fold the radial velocities. Default is False.

        Returns
        -------
        rvs : scalar or array
            Radial velocities [km/s]
        """
        # Calculating RVs for one period
        if fold is True:
            t = np.linspace(0, 1, nt)
            m_ano = 2 * np.pi * t
        else:
            t = np.linspace(self.t0, self.t0 + self.period, nt)   # Time (days)
            m_ano = 2 * np.pi / self.period * (t - self.t0)       # Mean anomaly
        e_ano = np.array([sp.newton(func=self.kep_eq, x0=mk, args=(mk,))
                          for mk in m_ano])                  # Eccentric anomaly
        # Computing the true anomaly
        f = 2 * np.arctan2(np.sqrt(1. + self.e) * np.sin(e_ano / 2),
                           np.sqrt(1. - self.e) * np.cos(e_ano / 2))
        # Why do we compute the true anomaly in this weird way? Because
        # arc-cosine is degenerate in the interval 0-360 degrees.

        rv = np.array([self.vr(fk) for fk in f])      # RVs (km/s)
        # Calculating RVs in the specified time interval
        if fold is True:
            rvs = np.interp(ts, t, rv, period=1)
        else:
            rvs = np.interp(ts, t, rv, period=self.period)
        return rvs


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    print('---------------------------------------')
    print('Starting test of keppy.orbit\n')
    t_sim = np.linspace(0, 1, 1000)
    start_time = time.time()  # We use this to measure the computation time

    # First, we create an instance of the system HIP156846
    HIP156846 = BinarySystem(log_k=np.log10(0.464),
                             log_period=np.log10(359.51),
                             t0=3998.1,
                             sqe_cosw=np.sqrt(0.847) * np.cos(np.radians(52.2)),
                             sqe_sinw=np.sqrt(0.847) * np.sin(np.radians(52.2)),
                             vz=-68.54)

    # The RVs are computed simply by running get_rvs()
    _rvs = HIP156846.get_rvs(nt=1000, ts=t_sim, fold=True)
    print('RV calculation took %.4f seconds' % (time.time() - start_time))

    # Plotting results
    plt.plot(t_sim, _rvs)
    plt.xlabel('JD - 2450000.0 (days)')
    plt.ylabel('RV (km/s)')
    plt.show()
