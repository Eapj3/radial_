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

    Parameters
    ----------
    t : sequence
        List of ``numpy.ndarray`` or a single ``numpy.ndarray`` object
        containing the time [JD - 2.4E6 days]

    rv : sequence
        List of ``numpy.ndarray`` or a single ``numpy.ndarray`` object
        containing the radial velocities [km/s]

    rv_err : sequence
        List of ``numpy.ndarray`` or a single ``numpy.ndarray`` object
        containing the uncertainties of the radial velocities [km/s]

    guess : sequence
        First guess of the orbital parameters in the following order: log10(K),
        log10(T), t0, sqrt(e)*cos(w) and sqrt(e)*sin(w).

    bounds_vz : sequence or ``tuple``
        Bounds for the estimation of proper motions of the barycenter (vz) for
        each dataset. It must have a `numpy.shape` equal to (n_datasets, 2), if
        n_datasets > 1. If n_datasets == 1, then its `numpy.shape` must be equal
        to (2,).

    bounds_sj: ``tuple`` or ``None``, optional
        Bounds for the estimation of the logarithm of the jitter noise for each
        dataset. It must have a `numpy.shape` equal to (n_datasets, 2), if
        n_datasets > 1. If n_datasets == 1, then its `numpy.shape` must be equal
        to (2,).

    bounds : ``tuple``, optional
        Bounds for the estimation of the orbital parameters, with the exception
        of the proper motion of the barycenter (vz). It must have numpy.shape
        equal to (5, 2). Default is ((-4, 4), (-4, 4), (0, 10000), (0, 360),
        (-4, -4.3E-5)).

    n_datasets : ``int``, optional
        Number of datasets to be used for the orbit estimation. Different
        datasets comprise, e.g., observations from different instruments. This
        is necessary because different instruments have different offsets in
        the radial velocities. Default is 1.

    fold: ``bool``, optional
        If True, the analysis will be performed by phase-folding the radial
        velocities. If False, analysis is performed on the given time array.
        Default is False.
    """
    def __init__(self, t, rv, rv_err, guess, bounds_vz, bounds_sj=None,
                 bounds=((-4, 4), (-4, 4), (0, 10000), (-1, 1), (-1, 1)),
                 n_datasets=1, fold=False):

        if isinstance(n_datasets, int) is False:
            raise TypeError('n_datasets must be int')
        elif n_datasets < 0:
            raise ValueError('n_datasets must be greater than zero')
        else:
            self.n_datasets = n_datasets

        # If user does not want to add extra noise term parameters, bounds_sj
        # must be set to ``None``.
        self.bounds_sj = bounds_sj
        if self.bounds_sj is None:
            if self.n_datasets == 1:
                self.t = t
                self.rv = rv
                self.rv_err = rv_err
                if len(guess) != 5 + self.n_datasets:
                    raise ValueError('guess must have a length equal to 5 + '
                                     'n_datasets')
                else:
                    self.guess = guess
                self.bounds = bounds + (bounds_vz,)
            else:
                self.t = t
                self.rv = rv
                self.rv_err = rv_err
                if len(guess) != 5 + self.n_datasets:
                    raise ValueError('guess must have a length equal to 5 + '
                                     'n_datasets')
                else:
                    self.guess = guess
                self.bounds = bounds + bounds_vz
        # If user wants to add extra noise term parameters, then bounds_sj must
        # not be ``None``.
        else:
            if self.n_datasets == 1:
                self.t = t
                self.rv = rv
                self.rv_err = rv_err
                if len(guess) != 5 + 2 * self.n_datasets:
                    raise ValueError('guess must have a length equal to 5 + '
                                     '2 * n_datasets')
                else:
                    self.guess = guess
                self.bounds = bounds + (bounds_vz,) + (bounds_sj,)
            else:
                self.t = t
                self.rv = rv
                self.rv_err = rv_err
                if len(guess) != 5 + 2 * self.n_datasets:
                    raise ValueError('guess must have a length equal to 5 + '
                                     '2 * n_datasets')
                else:
                    self.guess = guess
                self.bounds = bounds + bounds_vz + bounds_sj

        self.fold = fold

    # The likelihood function
    # noinspection PyTypeChecker
    def lnlike(self, theta):
        """
        This method produces the ln of the Gaussian likelihood function of a
        given set of parameters producing the observed data (t, rv +/- rv_err).

        Parameters
        ----------
        theta : ``numpy.ndarray``
            Array containing the 5+n_datasets parameters log_k, log_period, t0,
            w, log_e and the velocity offsets for each dataset

        Returns
        -------
        sum_like : ``float``
            The ln of the likelihood of the signal rv being the result of a
            model with parameters theta
        """
        # log_k, log_period, t0, w, log_e, vz = theta
        sum_like = 0
        if self.fold is True:
            time_array = self.t / (10 ** theta[1]) % 1
        else:
            time_array = self.t
        # Measuring the log-likelihood for each dataset separately
        for i in range(self.n_datasets):
            if self.n_datasets > 1:
                n = len(time_array[i])
            else:
                n = len(time_array[0])
            system = orbit.BinarySystem(log_k=theta[0], log_period=theta[1],
                                        t0=theta[2], sqe_cosw=theta[3],
                                        sqe_sinw=theta[4], vz=theta[5 + i])
            model = system.get_rvs(ts=time_array[i], nt=n)
            if self.bounds_sj is None:
                inv_sigma2 = 1. / (self.rv_err[i] ** 2)
            else:
                log_sigma_j = theta[5 + self.n_datasets + i]
                inv_sigma2 = 1. / (self.rv_err[i] ** 2 + (10 ** log_sigma_j)
                                   ** 2)
            sum_like += np.sum((self.rv[i] - model) ** 2 * inv_sigma2 +
                               np.log(2. * np.pi / inv_sigma2))
        sum_like *= -0.5
        return sum_like

    # Maximum likelihood estimation of orbital parameters
    def ml_orbit(self, maxiter=200, disp=False):
        """
        This method produces the maximum likelihood estimation of the orbital
        parameters.

        Parameters
        ----------
        maxiter : ``int``, optional
            Maximum number of iterations on scipy.minimize. Default=200

        disp : ``bool``, optional
            Display information about the minimization.

        Returns
        -------
        params : list
            An array with the estimated values of the parameters that best model
            the signal rv
        """
        nll = lambda *args: -self.lnlike(*args)
        result = op.minimize(fun=nll,
                             x0=self.guess,
                             method='TNC',
                             bounds=self.bounds,
                             options={'maxiter': maxiter, "disp": disp})

        if disp is True:
            print('Number of iterations performed = %i' % result['nit'])
            print('Minimization successful = %s' % repr(result['success']))
            print('Cause of termination = %s' % result['message'])

        params = result["x"]
        return params

    # Flat priors
    def flat(self, theta):
        """
        Computes a flat prior probability for a given set of parameters theta.

        Parameters
        ----------
        theta : sequence
            The orbital and instrumental parameters.

        Returns
        -------
        prob : ``float``
            The prior probability for a given set of orbital and instrumental
            parameters.
        """
        # Compute the eccentricity beforehand to impose a prior of e < 1 on it
        ecc = theta[3] ** 2 + theta[4] ** 2
        params = [self.bounds[i][0] < theta[i] < self.bounds[i][1]
                  for i in range(len(theta))]
        if all(params) is True and ecc < 1:
            prob = 0.0
        else:
            prob = -np.inf
        return prob

    # The probability
    def lnprob(self, theta):
        """
        This function calculates the ln of the probabilities to be used in the
        MCMC estimation.

        Parameters
        ----------
        theta: sequence
            The values of the orbital parameters log_k, log_period, t0, w, log_e

        Returns
        -------
        The probability of the signal rv being the result of a model with the
        parameters theta
        """
        lp = self.flat(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)

    # Using emcee to estimate the orbital parameters
    def emcee_orbit(self, nwalkers=20, nsteps=1000, nthreads=1, ballsizes=1E-2):
        """
        Calculates samples of parameters that best fit the signal rv.

        Parameters
        ----------
        nwalkers : ``int``
            Number of walkers

        nsteps : ``int``
            Number of burning-in steps

        nthreads : ``int``
            Number of threads in your machine

        ballsizes : scalar or sequence
            The one-dimensional size of the volume from which to generate a
            first position to start the chain.

        Returns
        -------
        sampler : ``emcee.EnsembleSampler``
            The resulting sampler object.
        """
        if self.bounds_sj is None:
            ndim = 5 + self.n_datasets
        else:
            ndim = 5 + 2 * self.n_datasets
        if isinstance(ballsizes, float) or isinstance(ballsizes, int):
            ballsizes = np.array([ballsizes] for i in range(ndim))
        pos = np.array([self.guess + ballsizes * np.random.randn(ndim)
                        for i in range(nwalkers)])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                        threads=nthreads)
        sampler.run_mcmc(pos, nsteps)
        return sampler
