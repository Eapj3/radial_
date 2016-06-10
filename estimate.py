#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as op
import orbit


# The likelihood function 
def lnlike(theta, t, rv, rv_err, vz, nt):
    """
    This function produces the ln of the Gaussian likelihood function of a 
    given set of parameters producing the observed data (t, rv +/- rv_err).
    
    theta = array containing the 6 parameters k, period, t0, w and e
    t = array of time [d]
    rv = array of radial velocities [km/s]
    rv_err = array of uncertainties in radial velocities [km/s]
    vz = proper motion [km/s]
    nt = number of points for one period
    """
    k, period, t0, w, e = theta
    model = orbit.get_rvs(k, period, t0, w, e, vz, nt, t)
    inv_sigma2 = 1./rv_err**2
    return -0.5*np.sum((rv-model)**2*inv_sigma2 + np.log(2.*np.pi/inv_sigma2))


# Maximum likelihood estimation of orbital parameters
def ml_orbit(t, rv, rv_err, guess, bnds, vz, nt):
    """
    This function produces the maximum likelihood estimation of the orbital
    parameters.
    
    t = array of time [d]
    rv = array of radial velocities [km/s]
    rv_err = array of uncertainties in radial velocities [km/s]
    guess = an array containing the first guesses of the parameters
    bnds = a sequence of tuples containing the bounds of the parameters
    vz = proper motion [km/s]
    nt = number of points for one period
    """
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(fun=nll,
                         x0=guess,
                         args=(t, rv, rv_err, vz, nt),
                         method='TNC',
                         bounds=bnds)
    k_ml, period_ml, t0_ml, w_ml, e_ml = result["x"]
    return [k_ml, period_ml, t0_ml, w_ml, e_ml]


# XXX Things below here are under development
"""
# emcee ######################

# Priors
def lnprior(theta):
    lnK, lnT, t0, w, lne, lna = theta
    if -5.0 < logK < 0.0 and \
       0.5 < logT < 1.5 and \
       1400 < t0 < 1600 and \
       0. < w < 20. and \
       -4 < loge < -0.5 and \
       -3 < lna < 0: 
        return -np.inf

# The probability
def lnprob(theta, x, y, yerr, VZ, NT):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, VZ, NT)

ndim, nwalkers = 6, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t_d, RV_d, RV_err,
                                VZ, NT))
sampler.run_mcmc(pos, 500)
"""
