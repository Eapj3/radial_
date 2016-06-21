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
    """This is an orbital parameter. Currently not in use by the code."""
    return m2/(m1+m2)*n*a*np.sin(i)/np.sqrt(1.-e**2)


# Calculates Eq. 65
def vr(vz, k, w, f, e):
    """
    The radial velocities equation.

    vz = Proper motion [km/s]
    k = Velocity semi-amplitude (Eq. 66) [km/s]
    w = Argument of periapse [degrees]
    f = True anomaly [rad]
    e = Eccentricity
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
    """
    return e_anom - e*np.sin(e_anom) - m_anom


# Calculates the radial velocities for given orbital parameters
def get_rvs(k, period, t0, w, e, vz, nt, ts):
    """
    Function that produces the radial velocities arrays given the following 
    parameters.

    k = Velocity semi-amplitude [km/s]
    period = orbital period [d]
    t0 = Time of periastron passage [d]
    w = Argument of periapse [degrees]
    e = eccentricity
    a = semi-major axis
    vz = proper motion [km/s]
    nt = number of points for one period
    ts = array of times [d]
    """

    # Calculating RVs for one period
    t = np.linspace(t0, t0+period, nt)                    # Time (days)
    m_anom = 2*np.pi/period*(t-t0)                             # Mean anomaly
    e_anom = np.array([sp.newton(func=kepler, x0 = mk, args = (e, mk)) \
                 for mk in m_anom])                       # Eccentric anomaly
    f = 2*np.arctan2(np.sqrt(1.+e)*np.sin(e_anom/2),
                     np.sqrt(1.-e)*np.cos(e_anom/2))      # True anomaly
    rv = np.array([vr(vz, k, w, fk, e) for fk in f]) # Radial velocities (km/s)

    # Calculating RVs in the specified time interval
    rvs = np.interp(ts, t, rv, period=period)
    return rvs


# Works the same as get_rvs, but the parameters that can't be negative are set
# in log-scale
def log_rvs(log_k, log_period, t0, w, log_e, vz, nt, ts):
    """
    Function that produces the radial velocities arrays given the following
    parameters.

    log_k = ln of the velocity semi-amplitude [km/s]
    log_period = ln of the period [d]
    t0 = Time of periastron passage [d]
    w = Argument of periapse [degrees]
    log_e = ln of the eccentricity
    vz = proper motion [km/s]
    nt = number of points for one period
    ts = array of times [d]
    """
    k = np.exp(log_k)
    period = np.exp(log_period)
    e = np.exp(log_e)
    return get_rvs(k, period, t0, w, e, vz, nt, ts)

