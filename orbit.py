#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import time

"""
This code is based on the formalism from Murray & Correia (2011), available
freely at http://arxiv.org/abs/1009.1738. The equation numbers are from this
article, unless otherwise noted.
"""

# Calculates the orbital parameter K (Eq. 66)
def K(m1, m2, n, a, I, e):
    """This is an orbital parameter. Currently not in use by the code."""
    return m2/(m1+m2)*n*a*np.sin(I)/np.sqrt(1.-e**2)

# Calculates Eq. 65
def vr(VZ, K, w, f, e):
    """The radial velocities equation."""
    w *= np.pi/180.
    return VZ + K*(np.cos(w+f) + e*np.cos(w))

# Calculates the Kepler equation (Eq. 41)
def kepler(E, e, M):
    """The Kepler equation."""
    return E - e*np.sin(E) - M

# Calculates the radial velocities for given orbital parameters
def get_RVs(K, T, t0, w, e, a, VZ, NT, ts):
    """
    Function that produces the time and radial velocities arrays given the
    following parameters. Radial velocities may be mirrored.

    K = orbit parameter [km/s]
    T = period [d]
    t0 = Time of periastron passage [d]
    w = Argument of periapse [degrees]
    e = eccentricity
    a = semi-major axis
    VZ = proper motion [km/s]
    NT = number of points for one period
    ts = array of times [d]
    """

    # Calculating RVs for one period
    t = np.linspace(t0, t0+T, NT)           # Time (days)
    M = 2*np.pi/T*(t-t0)                    # Mean anomaly
    E = np.array([sp.newton(func = kepler, x0 = Mk, args = (e, Mk)) \
                 for Mk in M])              # Eccentric anomaly
    r = a*(1.-e*np.cos(E))                  # r coordinates
    f = np.arccos(((a*(1.-e**2)/r)-1.0)/e)  # True anomalies
    f[(NT-1)/2:] += 2*(np.pi-f[(NT-1)/2:])  # Shifting the second half of f
    #f[:(NT-1)/2] += 2*(np.pi-f[:(NT-1)/2]) # Shifting the first half of f

    RV = np.array([vr(VZ, K, w, fk, e) \
                    for fk in f])           # Radial velocities (km/s)

    # Calculating RVs in the specified time interval
    RVs = np.interp(ts, t, RV, period = T)
    return RVs

# Works the same as get_RVs, but the parameters that can't be negative are set
# in log-scale
def log_RVs(logK, logT, t0, w, loge, loga, VZ, NT, ts):
    """
    Function that produces the time and radial velocities arrays given the
    following parameters. Radial velocities may be mirrored.

    logK = ln of the orbit parameter K [km/s]
    logT = ln of the period [d]
    t0 = Time of periastron passage [d]
    w = Argument of periapse [degrees]
    loge = ln of the eccentricity
    loga = ln of the semi-major axis
    VZ = proper motion [km/s]
    NT = number of points for one period
    ts = array of times [d]
    """
    K = np.exp(logK)
    T = np.exp(logT)
    e = np.exp(loge)
    a = np.exp(loga)
    return get_RVs(K, T, t0, w, e, a, VZ, NT, ts)

# Usage example
def example():
    """Example using the parameters of the star HD 156846 and its planet
    HD 156846 b."""
    ts = np.linspace(3600., 4200., 1000)
    start_time = time.time()
    RVs = log_RVs(logK = np.log(0.464),
                  logT = np.log(359.51),
                  t0 = 3998.1,
                  w = 52.2,
                  loge = np.log(0.847),
                  loga = np.log(0.9930),
                  VZ = -68.54,
                  NT = 1000,
                  ts = ts)
    print('RV calculation took %.4f seconds' % (time.time()-start_time))

    plt.plot(ts, RVs)
    plt.xlabel('JD - 2450000.0 (days)')
    plt.ylabel('RV (km/s)')
    plt.show()
