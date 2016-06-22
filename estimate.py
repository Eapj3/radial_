#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as op
import orbit
import emcee


# The likelihood function 
def lnlike(theta, t, rv, rv_err, vz):
    """
    This function produces the ln of the Gaussian likelihood function of a 
    given set of parameters producing the observed data (t, rv +/- rv_err).
    
    theta = array containing the 6 parameters k, period, t0, w and e
    t = array of time [d]
    rv = array of radial velocities [km/s]
    rv_err = array of uncertainties in radial velocities [km/s]
    vz = proper motion [km/s]
    nt = number of points for one period. Default=1000
    """
    nt = len(t)
    k, period, t0, w, e = theta
    model = orbit.get_rvs(k, period, t0, w, e, vz, nt, t)
    # log_k, log_period, t0, w, log_e = theta
    # model = orbit.log_rvs(log_k, log_period, t0, w, log_e, vz, nt, t)
    inv_sigma2 = 1./rv_err**2
    return -0.5*np.sum((rv-model)**2*inv_sigma2 + np.log(2.*np.pi/inv_sigma2))


# Maximum likelihood estimation of orbital parameters
def ml_orbit(t, rv, rv_err, guess, vz, k_interval=30., t0_interval=100.,
             maxiter=200):
    """
    This function produces the maximum likelihood estimation of the orbital
    parameters.
    
    t = array of time [d]
    rv = array of radial velocities [km/s]
    rv_err = array of uncertainties in radial velocities [km/s]
    guess = an array containing the first guesses of the parameters
    vz = proper motion [km/s]
    nt = number of points for one period. Default = 1000
    maxiter = maximum number of iterations on scipy.minimize. Default = 200
    """
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(fun=nll,
                         x0=guess,
                         args=(t, rv, rv_err, vz),
                         method='TNC',
                         bounds=((0., guess[0]+k_interval),
                                 (0, 1E4),
                                 (guess[2]-t0_interval, guess[2]+t0_interval),
                                 (0, 360),
                                 (0, 0.999999)),
                         options={'maxiter': maxiter})
    return result["x"]


# XXX Things below here are under very active development

# emcee ######################
'''
# Priors
def auto_lnprior(theta, k_max, t0_min, t0_max):
    """
    This function semi-automatically produces flat priors for the orbital
    parameters. It's not completely automatic because the user still has to
    provide the upper limit for the velocity semi-amplitude and the lower and
    upper limits for the time of periapse passage

    :param theta: array with shape [1,5] containing the values of the orbital
    parameters log_k, log_period, t0, w, log_e
    :param k_max: upper limit of the velocity semi-amplitude [km/s]
    :param t0_min: lower limit of the time of periapse passage [JD-2.45E6 days]
    :param t0_max: upper limit of the time of periapse passage [JD-2.45E6 days]
    :return: zero if all parameters are inside the flat prior interval, -inf
    otherwise
    """
    log_k, log_period, t0, w, log_e = theta
    if np.log(0.0001) < log_k < np.log(k_max) and \
        0. < log_period < np.log(10000) and \
        t0_min < t0 < t0_max and \
        0. < w < 360. and \
        np.log(0.0001) < log_e < np.log(0.9999):
        return 0.0
    return -np.inf


# The probability
def lnprob(theta, x, y, yerr, vz, k_max, t0_min, t0_max):
    """

    :param theta:
    :param x:
    :param y:
    :param yerr:
    :param vz:
    :param k_max:
    :param t0_min:
    :param t0_max:
    :return:
    """
    lp = auto_lnprior(theta, k_max, t0_min, t0_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, vz)


# Using emcee to estimate the orbital parameters
def emcee_orbit(t, rv, rv_err, guess, vz, k_max=60., t0_min=0., t0_max=7500.,
                nwalkers=50, nsteps=200, ncut=50):
    """

    :param t:
    :param rv:
    :param rv_err:
    :param guess:
    :param vz:
    :param k_max:
    :param t0_min:
    :param t0_max:
    :param nwalkers:
    :param nsteps:
    :return:
    """
    ndim = 5
    pos = np.array([guess + 1e-3*np.random.randn(ndim) for i in range(nwalkers)])
    pos[:, 0] = np.log(pos[:, 0])
    pos[:, 1] = np.log(pos[:, 1])
    pos[:, 4] = np.log(pos[:, 4])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, rv, rv_err,
                                    vz, k_max, t0_min, t0_max))
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:, ncut:, :].reshape((-1, ndim))
    return samples
'''
