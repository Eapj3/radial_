#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as op
from keppy import orbit
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

    :param t: list of arrays
        Time [JD - 2.4E6 days]

    :param rv: list of arrays
        Radial velocities [km/s]

    :param rv_err: list of arrays
        Uncertainties of the radial velocities [km/s]

    :param guess: array

    :param bounds_vz: tuple
        Bounds for the estimation proper motions of the barycenter (vz) for each
        dataset. It must have a `numpy.shape` equal to (n_datasets, 2), if
        n_datasets > 1. If n_datasets == 1, then its `numpy.shape` must be equal
        to (2,).

    :param bounds: tuple, optional
        Bounds for the estimation of the orbital parameters, with the exception
        of the proper motion of the barycenter (vz). It must have numpy.shape
        equal to (5, 2). Default is ((-4, 4), (-4, 4), (0, 10000), (0, 360),
        (-4, -4.3E-5)).

    :param n_datasets: int, optional
        Number of datasets to be used for the orbit estimation. Different
        datasets comprise, e.g., observations from different instruments. This
        is necessary because different instruments have different offsets in
        the radial velocities. Default is 1.
    """
    def __init__(self, t, rv, rv_err, guess, bounds_vz,
                 bounds=((-4, 4), (-4, 4), (0, 10000), (0, 360),
                         (-4, -4.3E-5)), n_datasets=1):

        if isinstance(n_datasets, int) is False:
            raise TypeError('n_datasets must be int')
        elif n_datasets < 0:
            raise ValueError('n_datasets must be greater than zero')
        else:
            self.n_datasets = n_datasets

        if self.n_datasets == 1:
            self.t = t
            self.rv = rv
            self.rv_err = rv_err
            if len(guess) != 5+self.n_datasets:
                raise ValueError('guess must have a length equal to 5 + '
                                 'n_datasets')
            else:
                self.guess = guess
            self.bounds = bounds + bounds_vz
        else:
            self.t = t
            self.rv = rv
            self.rv_err = rv_err
            if len(guess) != 5+self.n_datasets:
                raise ValueError('guess must have a length equal to 5 + '
                                 'n_datasets')
            else:
                self.guess = guess
            self.bounds = bounds + bounds_vz

    # The likelihood function
    def lnlike(self, theta):
        """
        This method produces the ln of the Gaussian likelihood function of a
        given set of parameters producing the observed data (t, rv +/- rv_err).

        :param theta: array
            Array containing the 5+n_datasets parameters log_k, log_period, t0,
            w, log_e and the velocity offsets for each dataset

        :return: float
            The ln of the likelihood of the signal rv being the result of a
            model with parameters theta
        """
        nt = len(self.t)
        # log_k, log_period, t0, w, log_e, vz = theta
        sum_like = 0
        # Measuring the log-likelihood for each dataset separately
        for i in range(self.n_datasets):
            system = orbit.BinarySystem(theta[0], theta[1], theta[2], theta[3],
                                        theta[4], theta[5 + i])
            model = system.get_rvs(ts=self.t[i], nt=nt)
            inv_sigma2 = 1. / self.rv_err[i] ** 2
            sum_like += np.sum((self.rv[i] - model) ** 2 * inv_sigma2 +
                               np.log(2. * np.pi / inv_sigma2))
        return -0.5 * sum_like

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

    # Flat priors
    def flat(self, theta):
        """

        :param theta:
        :return:
        """
        tests = [self.bounds[i][0] < theta[i] < self.bounds[i][1]
                 for i in range(len(theta))]
        if all(tests) is True:
            return 0.0
        return -np.inf

    # The probability
    def lnprob(self, theta):
        """
        This function calculates the ln of the probabilities to be used in the
        MCMC estimation.

        :param theta: array
            Array with shape [1,5] containing the values of the orbital
            parameters log_k, log_period, t0, w, log_e

        :return: scalar
            The probability of the signal rv being the result of a model with
            the parameters theta
        """
        lp = self.flat(theta)
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
        ndim = 5 + self.n_datasets
        pos = np.array([self.guess + 1e-4 * np.random.randn(ndim)
                        for i in range(nwalkers)])
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                        threads=nthreads)
        sampler.run_mcmc(pos, nsteps)
        samples = sampler.chain[:, ncut:, :].reshape((-1, ndim))
        return samples
