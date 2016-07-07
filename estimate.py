#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as op
import kepler
import emcee

"""
This code contains routines to estimate the orbital parameters of a binary
system by means of maximum likelihood estimation or a Markov-Chain Monte Carlo
estimation using the code emcee. Before using it, it is highly recommended to
read the documentation of emcee in order to understand what are priors,
probabilities, sampling and other jargon. Check it out at
http://dan.iel.fm/emcee/current/
"""


class OrbitalParams(object):
    """
    """
    def __init__(self, t, rv, rv_err, vz):
        self.t = t
        self.rv = rv
        self.rv_err = rv_err
        self.vz = vz

# The likelihood function 
def lnlike(theta, t, rv, rv_err, vz):
    """
    This function produces the ln of the Gaussian likelihood function of a
    given set of parameters producing the observed data (t, rv +/- rv_err).

    :param theta:
        Array containing the 5 parameters k, period, t0, w and e

    :param t:
        Array of time [d]

    :param rv:
        Array of radial velocities [km/s]

    :param rv_err:
        Array of uncertainties in radial velocities [km/s]

    :param vz:
        Proper motion [km/s]

    :return:
        The ln of the likelihood of the signal rv being the result of a model
        with parameters theta

    """
    nt = len(t)
    log_k, log_period, t0, w, log_e = theta
    model = orbit.log_rvs(log_k, log_period, t0, w, log_e, vz, nt, t)
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

    :param t:
        Array of time [d]

    :param rv:
        Array of radial velocities [km/s]

    :param rv_err:
        Array of uncertainties in radial velocities [km/s]

    :param guess:
        An array containing the first guesses of the parameters

    :param vz:
        Proper motion [km/s]

    :param k_interval:
        Interval that sets the upper bound for the scipy.optimize.minimize()
        function around the values of K in guess

    :param t0_interval:
        Interval that sets the bounds for the scipy.optimize.minimize() function
        around the value of t0 in guess

    :param maxiter:
        Maximum number of iterations on scipy.minimize. Default=200

    :return:
        An array with the estimated values of the parameters that best model the
        signal rv

    """
    nll = lambda *args: -lnlike(*args)
    guess[0] = np.log(guess[0])  # lnlike() works on log-space for K, period
    guess[1] = np.log(guess[1])  # and e
    guess[4] = np.log(guess[4])
    result = op.minimize(fun=nll,
                         x0=guess,
                         args=(t, rv, rv_err, vz),
                         method='TNC',
                         bounds=((-4, np.log(guess[0]+k_interval)),
                                 (-4, 4),
                                 (guess[2]-t0_interval, guess[2]+t0_interval),
                                 (0, 360),
                                 (-4, -1E-4)),
                         options={'maxiter': maxiter})
    return result["x"]


# Generating priors for Markov-Chain Monte Carlo estimation
def auto_lnprior(theta, k_max, t0_min, t0_max):
    """
    This function semi-automatically produces flat priors for the orbital
    parameters. It's not completely automatic because the user still has to
    provide the upper limit for the velocity semi-amplitude and the lower and
    upper limits for the time of periapse passage

    :param theta:
        Array with shape [1,5] containing the values of the orbital parameters
        log_k, log_period, t0, w, log_e

    :param k_max:
        Upper limit of the velocity semi-amplitude [km/s]

    :param t0_min:
        Lower limit of the time of periapse passage [JD-2.45E6 days]

    :param t0_max:
        Upper limit of the time of periapse passage [JD-2.45E6 days]

    :return:
        Zero if all parameters are inside the flat prior interval, -inf
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
def lnprob(theta, t, rv, rv_err, vz, k_max, t0_min, t0_max):
    """
    This function calculates the ln of the probabilities to be used in the MCMC
    esitmation.

    :param theta:
        Array with shape [1,5] containing the values of the orbital parameters
        log_k, log_period, t0, w, log_e

    :param t:
        Array of time [d]

    :param rv:
        Array of radial velocities [km/s]

    :param rv_err:
        Array of uncertainties in radial velocities [km/s]

    :param vz:
        Proper motion [km/s]

    :param k_max:
        Upper limit of the velocity semi-amplitude [km/s]

    :param t0_min:
        Lower limit of the time of periapse passage [JD-2.45E6 days]

    :param t0_max:
        Upper limit of the time of periapse passage [JD-2.45E6 days]

    :return:
        The probability of the signal rv being the result of a model with the
        parameters theta

    """
    lp = auto_lnprior(theta, k_max, t0_min, t0_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, t, rv, rv_err, vz)


# Using emcee to estimate the orbital parameters
def emcee_orbit(t, rv, rv_err, guess, vz, k_max=100., t0_min=0., t0_max=7500.,
                nwalkers=20, nsteps=1000, ncut=50, nthreads=1):
    """
    Calculates samples of parameters that best fit the signal rv.

    :param t:
        Array of time [d]

    :param rv:
        Array of radial velocities [km/s]

    :param rv_err:
        Array of uncertainties in radial velocities [km/s]

    :param guess:
        An array containing the first guesses of the parameters

    :param vz:
        Proper motion [km/s]

    :param k_max:
        Upper limit of the velocity semi-amplitude [km/s]

    :param t0_min:
        Lower limit of the time of periapse passage [JD-2.45E6 days]

    :param t0_max:
        Upper limit of the time of periapse passage [JD-2.45E6 days]

    :param nwalkers:
        Number of walkers

    :param nsteps:
        Number of burning-in steps

    :param ncut:
        Number of steps to ignore in the beginning of the burning-in phase

    :param nthreads:
        Number of threads in your machine

    :return:
        emcee samples that can be used to make a triangle plot using the corner
        routine

    """
    ndim = 5
    pos = np.array([guess + 1e-3*np.random.randn(ndim) for i in range(nwalkers)])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, rv, rv_err,
                                    vz, k_max, t0_min, t0_max),
                                    threads=nthreads)
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:, ncut:, :].reshape((-1, ndim))
    return samples
