#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as sp

"""
This module computes the radial velocities of a massive object being orbited by
a secondary massive object, and is based on the formalism from Murray & Correia
(2011), available freely at http://arxiv.org/abs/1009.1738. The equation numbers
 re from this article, unless otherwise noted.
"""


class Orbit(object):
    """
    A class that computes the radial velocities given the orbital parameters of
    the binary system.
    """
    def __init__(self, k, period, t0, w, e, vz):
        """

        :param k:
        :param period:
        :param t0:
        :param w:
        :param e:
        :param vz:
        """
        self.k = k
        self.period = period
        self.t0 = t0
        self.w = w
        self.e = e
        self.vz = vz
        self.wrad = w * np.pi / 180.

    # Calculates Eq. 65
    def vr(self, f):
        """
        The radial velocities equation.

        :param f:
            True anomaly [radians]

        :return:
            Radial velocities [km/s]
        """
        return self.vz + self.k * (np.cos(self.wrad + f) + self.e *
                                   np.cos(self.wrad))

    # Calculates the Kepler equation (Eq. 41)
    def kep_eq(self, e_anom, m_anom):
        """
        The Kepler equation.

        :param e_anom:
            Eccentric anomaly [radians]

        :param m_anom:
            Mean anomaly [radians]

        :return:
            Value of E-e*sin(E)-M
        """
        return e_anom - self.e * np.sin(e_anom) - m_anom

    # Calculates the radial velocities for given orbital parameters
    def get_rvs(self, nt, ts):
        """
        Computes the radial velocity arrays given the orbital parameters.

        :param nt:
            Number of points for one period

        :param ts:
            Array of times [d]

        :return:
            Array of radial velocities [km/s]
        """

        # Calculating RVs for one period
        t = np.linspace(self.t0, self.t0 + self.period, nt)       # Time (days)
        m_anom = 2 * np.pi / self.period * (t - self.t0)          # Mean anomaly
        e_anom = np.array([sp.newton(func=self.kep_eq, x0=mk, args=(self.e, mk))
                           for mk in m_anom])                # Eccentric anomaly
        # Computing the true anomaly
        f = 2 * np.arctan2(np.sqrt(1. + self.e) * np.sin(e_anom / 2),
                           np.sqrt(1. - self.e) * np.cos(e_anom / 2))
        # Why do we compute the true anomaly in this weird way? Because
        # arc-cosine is degenerate in the interval 0-360 degrees.
        rv = np.array([self.vr(fk) for fk in f])      # RVs (km/s)

        # Calculating RVs in the specified time interval
        rvs = np.interp(ts, t, rv, period=self.period)
        return rvs


# Works the same as get_rvs, but the parameters that cannot be negative are set
# in log-scale
def log_rvs(log_k, log_period, t0, w, log_e, vz, nt, ts):
    """
    Function that produces the radial velocities arrays given the following
    parameters.

    :param log_k:
        ln of velocity semi-amplitude [km/s]

    :param log_period:
        ln of orbital period [days]

    :param t0:
        Time of periastron passage [days]

    :param w:
        Argument of periapse [degrees]

    :param log_e:
        ln of eccentricity of the orbit

    :param vz:
        Proper motion [km/s]

    :param nt:
        Number of points for one period

    :param ts:
        Array of times [d]

    :return:
        Array of radial velocities [km/s]

    """
    k = np.exp(log_k)
    period = np.exp(log_period)
    e = np.exp(log_e)
    return get_rvs(k, period, t0, w, e, vz, nt, ts)
