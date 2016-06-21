#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as op
import orbit
#import emcee


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
def ml_orbit(t, rv, rv_err, guess, bnds, vz, nt, maxiter=200):
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
    maxiter = maximum number of iterations on scipy.minimize. Default = 200
    """
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(fun=nll,
                         x0=guess,
                         args=(t, rv, rv_err, vz, nt),
                         method='TNC',
                         bounds=bnds,
                         options={'maxiter': maxiter})
    return result["x"]


# XXX Things below here are under development
'''
# emcee ######################

# Priors
def auto_lnprior(theta, k_max=60., t0_min=0., t0_max=7500.):
    """
    This function semi-automatically produces flat priors for the orbital
    parameters. It's not completely automatic because the user still has to
    provide the upper limit for the velocity semi-amplitude and the lower and
    upper limits for the time of periapse passage

    :param theta: array with shape [1,5] containing the values of the orbital
    parameters log_k, log_period, t0, w, log_e
    :param k_max: upper limit of the velocity semi-amplitude [km/s], default=60.
    :param t0_min: lower limit of the time of periapse passage [JD-2.45E6 days],
    default=0.
    :param t0_max: upper limit of the time of periapse passage [JD-2.45E6 days],
    default=7500.
    :return: zero if all parameters are inside the flat prior interval, -inf
    otherwise
    """
    log_k, log_period, t0, w, log_e = theta
    if -3.0 < log_k < np.log10(k_max) and \
        0. < log_period < 4. and \
        t0_min < t0 < t0_max and \
        0. < w < 360. and \
       -3 < log_e < -0.001:
        return 0.0
    return -np.inf


# The probability
def lnprob(theta, x, y, yerr, vz, nt):
    lp = auto_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, vz, nt)

ndim, nwalkers = 5, 100

pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t_d, RV_d, RV_err,
                                VZ, NT))
sampler.run_mcmc(pos, 500)
'''