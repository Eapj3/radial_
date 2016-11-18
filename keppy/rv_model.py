#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keppy import orbit

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
    t : ``astropy.units.Quantity``
        Time.

    log_k : ``astropy.units.Quantity``
        Base-10 logarithm of the radial velocity semi-amplitude
        (dex(velocity unit)).

    log_period : ``astropy.units.Quantity``
        Base-10 logarithm of the orbital period (dex(time unit)).

    t0 : ``astropy.units.Quantity``
        Time of pariastron passage (time unit).

    omega : ``astropy.units.Quantity``
        Argument of periapse (angle unit).

    log_ecc : scalar
        Base-10 logarithm of the eccentricity of the orbit.

    gamma : ``astropy.units.Quantity``
        Proper motion of the barycenter (velocity unit).

    Returns
    -------
    rvs : ``astropy.units.Quantity``
        Radial velocity.
    """
    try:
        k = log_k.physical
    except AttributeError:
        k = 10 ** log_k

    try:
        period = log_period.physical
    except AttributeError:
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
    t : ``astropy.units.Quantity``
        Time.

    log_k : ``astropy.units.Quantity``
        Base-10 logarithm of the radial velocity semi-amplitude
        (dex(velocity unit)).

    log_period : ``astropy.units.Quantity``
        Base-10 logarithm of the orbital period (dex(time unit)).

    t0 : ``astropy.units.Quantity``
        Time of pariastron passage (time unit).

    sqe_cosw : scalar
        sqrt(ecc) * cos(omega).

    sqe_sinw : scalar
        sqrt(ecc) * sin(omega).

    gamma : ``astropy.units.Quantity``
        Proper motion of the barycenter (velocity unit).

    Returns
    -------
    rvs : ``astropy.units.Quantity``
        Radial velocity.
    """
    try:
        k = log_k.physical
    except AttributeError:
        k = 10 ** log_k

    try:
        period = log_period.physical
    except AttributeError:
        period = 10 ** log_period

    system = orbit.BinarySystem(k, period, t0, sqe_cosw=sqe_cosw,
                                sqe_sinw=sqe_sinw, gamma=gamma)
    rvs = system.get_rvs(t)
    return rvs
