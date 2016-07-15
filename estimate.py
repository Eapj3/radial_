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
    A class that computes the orbital parameters of a binary system given its
    radial velocities (and their uncertainties) in function of time.

    :param t: array
        Time [JD - 2.4E6 days]

    :param rv: array
        Radial velocities [km/s]

    :param rv_err: array
        Uncertainties of the radial velocities [km/s]

    :param guess: array


    :param bounds: tuple


    :param vz: scalar
        Proper motion [km/s]
    """
    def __init__(self, t, rv, rv_err, vz, guess,
                 bounds=((-4, 4), (-4, 4), (0, 10000), (0, 360),
                         (-4, -4.3E-5)), prior=None, args=None):
        self.t = t
        self.rv = rv
        self.rv_err = rv_err
        self.vz = vz
        self.guess = guess
        self.bounds = bounds
        self.prior = prior
        self.args = args

    # The likelihood function
    def lnlike(self, theta):
        """
        This method produces the ln of the Gaussian likelihood function of a
        given set of parameters producing the observed data (t, rv +/- rv_err).

        :param theta: array
            Array containing the 5 parameters log_k, log_period, t0, w and log_e

        :return: float
            The ln of the likelihood of the signal rv being the result of a
            model with parameters theta
        """
        nt = len(self.t)
        log_k, log_period, t0, w, log_e = theta
        system = kepler.BinarySystem(log_k, log_period, t0, w, log_e, self.vz)
        model = system.get_rvs(ts=self.t, nt=nt)
        inv_sigma2 = 1. / self.rv_err ** 2
        return -0.5 * np.sum((self.rv - model) ** 2 * inv_sigma2 +
                             np.log(2. * np.pi / inv_sigma2))

    # Maximum likelihood estimation of orbital parameters
    def ml_orbit(self, maxiter=200):
        """
        This method produces the maximum likelihood estimation of the orbital
        parameters.

        :param maxiter: int
            Maximum number of iterations on scipy.minimize. Default=200

        :return: array
            An array with the estimated values of the parameters that best model
            the signal rv

        """
        nll = lambda *args: -self.lnlike(*args)
        result = op.minimize(fun=nll,
                             x0=self.guess,
                             method='TNC',
                             bounds=self.bounds,
                             options={'maxiter': maxiter})
        return result["x"]

    # The probability
    def lnprob(self, theta):
        """
        This function calculates the ln of the probabilities to be used in the
        MCMC esitmation.

        :param theta: array
            Array with shape [1,5] containing the values of the orbital
            parameters log_k, log_period, t0, w, log_e

        :return: scalar
            The probability of the signal rv being the result of a model with
            the parameters theta
        """
        lp = self.prior(*self.args)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)

    # Using emcee to estimate the orbital parameters
    def emcee_orbit(self, nwalkers=20, nsteps=1000, ncut=50, nthreads=1):
        """
        Calculates samples of parameters that best fit the signal rv.

        :param nwalkers: int
            Number of walkers

        :param nsteps: int
            Number of burning-in steps

        :param ncut: int
            Number of steps to ignore in the beginning of the burning-in phase

        :param nthreads: int
            Number of threads in your machine

        :return: array
            emcee samples that can be used to make a triangle plot using the
            corner routine
        """
        ndim = 5
        pos = np.array([self.guess + 1e-3 * np.random.randn(ndim)
                        for i in range(nwalkers)])
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                        threads=nthreads)
        sampler.run_mcmc(pos, nsteps)
        samples = sampler.chain[:, ncut:, :].reshape((-1, ndim))
        return samples


class Priors(object):
    """

    :param theta:
    """
    def __init__(self, theta):
        self.theta = theta

    # Flat priors
    def flat(self, bounds):
        """

        :param bounds: array or list
        :return:
        """
        log_k, log_period, t0, w, log_e = self.theta
        if bounds[0][0] < log_k < bounds[0][1] and \
           bounds[1][0] < log_period < bounds[1][1] and \
           bounds[2][0] < t0 < bounds[2][1] and \
           bounds[3][0] < w < bounds[3][1] and \
           bounds[4][0] < log_e < bounds[4][1]:
            return 0.0
        return -np.inf
