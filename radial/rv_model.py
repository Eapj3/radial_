#!/usr/bin/env python
# -*- coding: utf-8 -*-

from radial import orbit

"""
This module contains the different radial velocities models currently available
with keppy.
"""


# The RV model from Murray & Correia 2010
def mc10(t, log_k, log_period, t0, omega, log_ecc, gamma):
    """
    The radial velocities model from Murray & Correia 2010.

    Parameters
    ----------
    t : scalar
        Time in days.

    log_k : scalar
        Base-10 logarithm of the radial velocity semi-amplitude in dex(m / s).

    log_period : scalar
        Base-10 logarithm of the orbital period in dex(d).

    t0 : scalar
        Time of pariastron passage in days.

    omega : scalar
        Argument of periapse in radians.

    log_ecc : scalar
        Base-10 logarithm of the eccentricity of the orbit.

    gamma : scalar
        Instrumental radial velocity offset in m / s.

    Returns
    -------
    rvs : scalar
        Radial velocity in m / s.
    """
    k = 10 ** log_k
    period = 10 ** log_period
    ecc = 10 ** log_ecc
    system = orbit.BinarySystem(k, period, t0, omega, ecc, gamma=gamma)
    rvs = system.get_rvs(t)
    return rvs


# The RV model from EXOFAST
def exofast(t, log_k, log_period, t0, sqe_cosw, sqe_sinw, gamma):
    """
    The radial velocities model from EXOFAST (Eastman et al. 2013).

    Parameters
    ----------
    t : scalar
        Time in days.

    log_k : scalar
        Base-10 logarithm of the radial velocity semi-amplitude in dex(m / s).

    log_period : scalar
        Base-10 logarithm of the orbital period in dex(d).

    t0 : scalar
        Time of pariastron passage in days.

    sqe_cosw : scalar
        sqrt(ecc) * cos(omega).

    sqe_sinw : scalar
        sqrt(ecc) * sin(omega).

    gamma : scalar
        Instrumental radial velocity offset in m / s.

    Returns
    -------
    rvs : scalar
        Radial velocity in m / s.
    """
    k = 10 ** log_k
    period = 10 ** log_period
    system = orbit.BinarySystem(k, period, t0, sqe_cosw=sqe_cosw,
                                sqe_sinw=sqe_sinw, gamma=gamma)
    rvs = system.get_rvs(t)
    return rvs
