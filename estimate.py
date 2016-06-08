#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as op
#import emcee
#import corner
import orbit

# The likelihood function 
def lnlike(theta, t, RV, RVerr, VZ, NT):
    """
    This function produces the ln of the Gaussian likelihood function of a 
    given set of parameters producing the observed data (t, RV +/- RVerr). 
    
    theta = array containing the 6 parameters K, T, t0, w, e and a
    t = array of time [d]
    RV = array of radial velocities [km/s]
    RVerr = array of uncertainties in radial velocities [km/s]
    VZ = proper motion [km/s]
    NT = number of points for one period
    """
    K, T, t0, w, e, a = theta
    model = orbit.get_rvs(K, T, t0, w, e, a, VZ, NT, t)
    inv_sigma2 = 1./RVerr**2
    return -0.5*np.sum((RV-model)**2*inv_sigma2 + np.log(2.*np.pi/inv_sigma2))

# Maximum likelihood estimation of orbital parameters
def ml_orbit(t, RV, RVerr, guess, bnds, VZ, NT):
    """
    This function produces the maximum likelihood estimation of the orbital
    parameters.
    
    t = array of time [d]
    RV = array of radial velocities [km/s]
    RVerr = array of uncertainties in radial velocities [km/s]
    guess = an array containing the first guesses of the parameters
    bnds = a sequence of tuples containing the bounds of the parameters
    VZ = proper motion [km/s]
    NT = number of points for one period
    """
    #x0 = [K_true, T_true, t0_true, w_true, e_true, a_true], 
    #bnds = ((0, 1),(300, 400),(3600, 4200),(0, 360),(0, 1),(0, 5))
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(fun = nll, 
                         x0 = guess,
                         args=(t, RV, RVerr, VZ, NT),
                         method = 'TNC',
                         bounds = bnds)
    K_ml, T_ml, t0_ml, w_ml, e_ml, a_ml = result["x"]
    return [K_ml, T_ml, t0_ml, w_ml, e_ml, a_ml]

# XXX Things below here are under development
"""
# emcee ######################3

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
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t_d, RV_d, RV_err, VZ, NT))
sampler.run_mcmc(pos, 500)
"""
