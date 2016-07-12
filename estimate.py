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
    """
    def __init__(self, t, rv, rv_err, vz):
        self.t = t
        self.rv = rv
        self.rv_err = rv_err
        self.vz = vz

    # The likelihood function
    def lnlike(self, theta):
        """
        This method produces the ln of the Gaussian likelihood function of a
        given set of parameters producing the observed data (t, rv +/- rv_err).

        :param theta:
            Array containing the 5 parameters log_k, log_period, t0, w and log_e

        :return:
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
    def ml_orbit(self, guess, log_k_interval=2., t0_interval=100., maxiter=200):
        """
        This method produces the maximum likelihood estimation of the orbital
        parameters.

        :param guess:
            An array containing the first guesses of the parameters

        :param log_k_interval:
            Interval that sets the upper bound for the scipy.optimize.minimize()
            function around the values of log10(K) in guess

        :param t0_interval:
            Interval that sets the bounds for the scipy.optimize.minimize()
            function around the value of t0 in guess

        :param maxiter:
            Maximum number of iterations on scipy.minimize. Default=200

        :return:
            An array with the estimated values of the parameters that best model
            the signal rv

        """
        nll = lambda *args: -self.lnlike(*args)
        result = op.minimize(fun=nll,
                             x0=guess,
                             method='TNC',
                             bounds=((-4, guess[0] + log_k_interval),
                                     (-4, 4),
                                     (guess[2] - t0_interval,
                                      guess[2] + t0_interval),
                                     (0, 360),
                                     (-4, -0.0001)),
                             options={'maxiter': maxiter})
        return result["x"]

    # Generating priors for Markov-Chain Monte Carlo estimation
    @staticmethod
    def auto_lnprior(theta, log_k_max, t0_min, t0_max):
        """
        This method semi-automatically produces flat priors for the orbital
        parameters. It's not completely automatic because the user still has to
        provide the upper limit for the velocity semi-amplitude and the lower
        and upper limits for the time of periapse passage

        :param theta:
            Array with shape [1,5] containing the values of the orbital
            parameters log_k, log_period, t0, w, log_e

        :param log_k_max:
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
        if -4 < log_k < log_k_max and \
           0. < log_period < 4 and \
           t0_min < t0 < t0_max and \
           0. < w < 360. and \
           -4 < log_e < -0.0001:
            return 0.0
        return -np.inf

    # The probability
    def lnprob(self, theta, log_k_max, t0_min, t0_max):
        """
        This function calculates the ln of the probabilities to be used in the
        MCMC esitmation.

        :param theta:
            Array with shape [1,5] containing the values of the orbital
            parameters log_k, log_period, t0, w, log_e

        :param log_k_max:
            Upper limit of the velocity semi-amplitude [km/s]

        :param t0_min:
            Lower limit of the time of periapse passage [JD-2.45E6 days]

        :param t0_max:
            Upper limit of the time of periapse passage [JD-2.45E6 days]

        :return:
            The probability of the signal rv being the result of a model with the
            parameters theta
        """
        lp = self.auto_lnprior(theta, log_k_max, t0_min, t0_max)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)

    # Using emcee to estimate the orbital parameters
    def emcee_orbit(self, guess, log_k_max=2., t0_min=0., t0_max=7500.,
                    nwalkers=20, nsteps=1000, ncut=50, nthreads=1):
        """
        Calculates samples of parameters that best fit the signal rv.

        :param guess:
            An array containing the first guesses of the parameters

        :param log_k_max:
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
            emcee samples that can be used to make a triangle plot using the
            corner routine
        """
        ndim = 5
        pos = np.array([guess + 1e-3 * np.random.randn(ndim)
                        for i in range(nwalkers)])
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                        args=(log_k_max, t0_min, t0_max),
                                        threads=nthreads)
        sampler.run_mcmc(pos, nsteps)
        samples = sampler.chain[:, ncut:, :].reshape((-1, ndim))
        return samples
