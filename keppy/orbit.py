#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import scipy.optimize as sp
from astropy import units as u

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

    w : scalar
        Argument of periapse [degrees]

    log_e : scalar
        Natural logarithm of the eccentricity

    vz : scalar
        Proper motion of the barycenter [km/s]
    """
    def __init__(self, log_k, log_period, t0, w, log_e, vz, debug=False):

        if isinstance(log_k, float) or isinstance(log_k, int):
            self.log_k = log_k
        else:
            raise TypeError('log_k is not scalar')

        if isinstance(log_period, float) or isinstance(log_period, int):
            self.log_period = log_period
        else:
            raise TypeError('log_period is not scalar')

        if isinstance(t0, float) or isinstance(t0, int):
            self.t0 = t0
        else:
            raise TypeError('t0 is not scalar')

        if isinstance(w, float) or isinstance(w, int):
            self.w = w
        else:
            raise TypeError('w is not scalar')

        if isinstance(log_e, float) or isinstance(log_e, int):
            self.log_e = log_e
        else:
            raise TypeError('log_e is not scalar')

        if isinstance(vz, float) or isinstance(vz, int):
            self.vz = vz
        else:
            raise TypeError('vz is not scalar')

        if isinstance(debug, bool):
            self.debug_flag = debug
        else:
            raise TypeError('debug must be boolean')

        self.w_rad = w * np.pi / 180.
        self.k = 10 ** log_k
        self.period = 10 ** log_period
        self.e = 10 ** log_e

        if self.e > 0.9999:
            raise ValueError('Keplerian orbits are ellipses, therefore e <= 1')

    # Calculates Eq. 65
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

        :param e_ano: scalar or array
            Eccentric anomaly [radians]

        :param m_ano: scalar or array
            Mean anomaly [radians]

        :return: scalar or array
            Value of E-e*sin(E)-M
        """
        return e_ano - self.e * np.sin(e_ano) - m_ano

    # Calculates the radial velocities for given orbital parameters
    def get_rvs(self, ts, nt=1000):
        """
        Computes the radial velocity given the orbital parameters.

        :param ts: scalar or array
            Time [d]

        :param nt: int
            Number of points for one period. Default=1000.

        :return: scalar or array
            Radial velocities [km/s]
        """
        # Calculating RVs for one period
        t = np.linspace(self.t0, self.t0 + self.period, nt)       # Time (days)
        m_ano = 2 * np.pi / self.period * (t - self.t0)           # Mean anomaly
        e_ano = np.array([sp.newton(func=self.kep_eq, x0=mk, args=(mk,))
                          for mk in m_ano])                  # Eccentric anomaly
        # Computing the true anomaly
        f = 2 * np.arctan2(np.sqrt(1. + self.e) * np.sin(e_ano / 2),
                           np.sqrt(1. - self.e) * np.cos(e_ano / 2))
        # Why do we compute the true anomaly in this weird way? Because
        # arc-cosine is degenerate in the interval 0-360 degrees.

        if self.debug_flag is True:
            print('Median of mean anomalies = %.3f' % (np.median(m_ano)))
            print('Median of eccentric anomalies = %.3f' % (np.median(e_ano)))
            print('Median of true anomalies = %.3f' % (np.median(f)))
            print('Std deviation of mean anomalies = %.3f' % (np.std(m_ano)))
            print('Std deviation of eccentric anomalies = %.3f' %
                  (np.std(e_ano)))
            print('Std deviation of true anomalies = %.3f\n' % (np.std(f)))

        rv = np.array([self.vr(fk) for fk in f])      # RVs (km/s)
        # Calculating RVs in the specified time interval
        rvs = np.interp(ts, t, rv, period=self.period)
        return rvs


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    print('---------------------------------------')
    print('Starting test of keppy.orbit\n')
    t_sim = np.linspace(3600., 4200., 1000)  # The time window [JD-2.45E6 days]
    start_time = time.time()  # We use this to measure the computation time

    # First, we create an instance of the system HIP156846
    HIP156846 = BinarySystem(log_k=np.log10(0.464),
                             log_period=np.log10(359.51),
                             t0=3998.1,
                             w=52.2,
                             log_e=np.log10(1.847),
                             vz=-68.54,
                             debug=True)

    # The RVs are computed simply by running get_rvs()
    _rvs = HIP156846.get_rvs(nt=1000, ts=t_sim)
    print('RV calculation took %.4f seconds' % (time.time() - start_time))

    # Plotting results
    plt.plot(t_sim, _rvs)
    plt.xlabel('JD - 2450000.0 (days)')
    plt.ylabel('RV (km/s)')
    plt.show()
