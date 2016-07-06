#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as sp

"""
This code is based on the formalism from Murray & Correia (2011), available
freely at http://arxiv.org/abs/1009.1738. The equation numbers are from this
article, unless otherwise noted.
"""


# Calculates the orbital parameter K (Eq. 66) -- velocity semi-amplitude
def k_orb(m1, m2, n, a, i, e):
    """
    This is an orbital parameter. Currently not in use by the code.

    :param m1:
        Mass of body 1

    :param m2:
        Mass of body 2

    :param n:
        Mean motion of the orbiting body 2 (2*pi/period)

    :param a:
        Semi-major axis of the orbiting body 2

    :param i:
        Inclination of the orbital plane [radians]

    :param e:
        Eccentricity of the orbit

    :return:
        The velocity semi-amplitude

    """
    return m2/(m1+m2)*n*a*np.sin(i)/np.sqrt(1.-e**2)


# Calculates Eq. 65
def vr(vz, k, w, f, e):
    """
    The radial velocities equation.

    :param vz:
        Proper motion [km/s]

    :param k:
        Velocity semi-amplitude [km/s]

    :param w:
        Argument of periapse [degrees]

    :param f:
        True anomaly [radians]

    :param e:
        Eccentricity of the orbit

    :return:
        Radial velocities [km/s]

    """
    w *= np.pi/180.
    return vz + k*(np.cos(w+f) + e*np.cos(w))


# Calculates the Kepler equation (Eq. 41)
def kepler(e_anom, e, m_anom):
    """
    The Kepler equation.

    e_anom = eccentric anomaly [rad]
    e = eccentricity
    m_anom = mean anomaly [rad]
    :param e_anom:
        Eccentric anomaly [radians]

    :param e:
        Eccentricity of the orbit

    :param m_anom:
        Mean anomaly [radians]

    :return:
        Value of E-e*sin(E)-M

    """
    return e_anom - e*np.sin(e_anom) - m_anom


# Calculates the radial velocities for given orbital parameters
def get_rvs(k, period, t0, w, e, vz, nt, ts):
    """
    Function that produces the radial velocities arrays given the following
    parameters.

    :param k:
        Velocity semi-amplitude [km/s]

    :param period:
        Orbital period [days]

    :param t0:
        Time of periastron passage [days]

    :param w:
        Argument of periapse [degrees]

    :param e:
        Eccentricity of the orbit

    :param vz:
        Proper motion [km/s]

    :param nt:
        Number of points for one period

    :param ts:
        Array of times [d]

    :return:
        Array of radial velocities [km/s]

    """

    # Calculating RVs for one period
    t = np.linspace(t0, t0+period, nt)                    # Time (days)
    m_anom = 2*np.pi/period*(t-t0)                        # Mean anomaly
    e_anom = np.array([sp.newton(func=kepler, x0=mk, args=(e, mk))
                       for mk in m_anom])                 # Eccentric anomaly
    f = 2*np.arctan2(np.sqrt(1.+e)*np.sin(e_anom/2),
                     np.sqrt(1.-e)*np.cos(e_anom/2))      # True anomaly
    # Why do we compute the true anomaly in this weird way? Because arc-cosine
    # is degenerate in the interval 0-360 degrees.
    rv = np.array([vr(vz, k, w, fk, e) for fk in f])      # RVs (km/s)

    # Calculating RVs in the specified time interval
    rvs = np.interp(ts, t, rv, period=period)
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
