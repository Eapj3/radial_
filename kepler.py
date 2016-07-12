#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    """
    def __init__(self, log_k, log_period, t0, w, log_e, vz):
        """

        :param log_k:
        :param log_period:
        :param t0:
        :param w:
        :param log_e:
        :param vz:
        """
        self.log_k = log_k
        self.log_period = log_period
        self.t0 = t0
        self.w = w
        self.log_e = log_e
        self.vz = vz
        self.w_rad = w * np.pi / 180.
        self.k = 10 ** log_k
        self.period = 10 ** log_period
        self.e = 10 ** log_e

    # Calculates Eq. 65
    def vr(self, f):
        """
        The radial velocities equation.

        :param f:
            True anomaly [radians]

        :return:
            Radial velocities [km/s]
        """
        return self.vz + self.k * (np.cos(self.w_rad + f) + self.e *
                                   np.cos(self.w_rad))

    # Calculates the Kepler equation (Eq. 41)
    def kep_eq(self, e_ano, m_ano):
        """
        The Kepler equation.

        :param e_ano:
            Eccentric anomaly [radians]

        :param m_ano:
            Mean anomaly [radians]

        :return:
            Value of E-e*sin(E)-M
        """
        return e_ano - self.e * np.sin(e_ano) - m_ano

    # Calculates the radial velocities for given orbital parameters
    def get_rvs(self, ts, nt=1000):
        """
        Computes the radial velocity arrays given the orbital parameters.

        :param ts:
            Array of times [d]

        :param nt:
            Number of points for one period. Default=1000.

        :return:
            Array of radial velocities [km/s]
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
        rv = np.array([self.vr(fk) for fk in f])      # RVs (km/s)
        # Calculating RVs in the specified time interval
        rvs = np.interp(ts, t, rv, period=self.period)
        return rvs
