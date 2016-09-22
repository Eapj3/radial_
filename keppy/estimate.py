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
        First guess of the orbital parameters in the following order: log10(K),
        log10(T), t0, w and log10(e).

    :param bounds_vz: tuple
        Bounds for the estimation of proper motions of the barycenter (vz) for
        each dataset. It must have a `numpy.shape` equal to (n_datasets, 2), if
        n_datasets > 1. If n_datasets == 1, then its `numpy.shape` must be equal
        to (2,).

    :param bounds_sj: tuple
        Bounds for the estimation of jitter noise for each dataset. It must have
        a `numpy.shape` equal to (n_datasets, 2), if n_datasets > 1. If
        n_datasets == 1, then its `numpy.shape` must be equal to (2,).

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
    def __init__(self, t, rv, rv_err, guess, bounds_vz, bounds_sj,
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

    # The likelihood function
    # noinspection PyTypeChecker
    def lnlike(self, theta):
        """
        This method produces the ln of the Gaussian likelihood function of a
        given set of parameters producing the observed data (t, rv +/- rv_err).

        :param theta: array
            Array containing the 5+n_datasets parameters log_k, log_period, t0,
            w, log_e and the velocity offsets for each dataset

        :return sum_like: float
            The ln of the likelihood of the signal rv being the result of a
            model with parameters theta
        """
        # log_k, log_period, t0, w, log_e, vz = theta
        sum_like = 0
        # Measuring the log-likelihood for each dataset separately
        if self.n_datasets == 1:
            n = len(self.t)
            sigma_j = theta[5 + self.n_datasets]
            system = orbit.BinarySystem(log_k=theta[0], log_period=theta[1],
                                        t0=theta[2], w=theta[3],
                                        log_e=theta[4], vz=theta[5])
            model = system.get_rvs(ts=self.t, nt=n)
            inv_sigma2 = 1. / (self.rv_err ** 2 + sigma_j ** 2)
            sum_like = 0.5 * np.sum((self.rv - model) ** 2 * inv_sigma2 +
                                    np.log(2. * np.pi / inv_sigma2))
        else:
            for i in range(self.n_datasets):
                n = len(self.t[i])
                sigma_j = theta[5 + self.n_datasets + i]
                system = orbit.BinarySystem(log_k=theta[0], log_period=theta[1],
                                            t0=theta[2], w=theta[3],
                                            log_e=theta[4], vz=theta[5 + i])
                model = system.get_rvs(ts=self.t[i], nt=n)
                inv_sigma2 = 1. / (self.rv_err[i] ** 2 + sigma_j ** 2)
                sum_like += np.sum((self.rv[i] - model) ** 2 * inv_sigma2 +
                                   np.log(2. * np.pi / inv_sigma2))
            sum_like *= -0.5
        return sum_like

    # Maximum likelihood estimation of orbital parameters
    def ml_orbit(self, maxiter=200, disp=False):
        """
        This method produces the maximum likelihood estimation of the orbital
        parameters.

        :param maxiter: int, optional
            Maximum number of iterations on scipy.minimize. Default=200

        :param disp: bool, optional
            Display information about the minimization.

        :return params: array
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

        :param theta: array
            Array containing the 5+n_datasets parameters log_k, log_period, t0,
            w, log_e and the velocity offsets for each dataset

        :return prob:
            The prior probability for a given set of orbital parameters.
        """
        params = [self.bounds[i][0] < theta[i] < self.bounds[i][1]
                  for i in range(len(theta))]
        if all(params) is True:
            prob = 0.0
        else:
            prob = -np.inf
        return prob

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
    def emcee_orbit(self, nwalkers=20, nsteps=1000, nthreads=1, ballsize=1E-2):
        """
        Calculates samples of parameters that best fit the signal rv.

        :param nwalkers: int
            Number of walkers

        :param nsteps: int
            Number of burning-in steps

        :param nthreads: int
            Number of threads in your machine

        :param ballsize: float
            The one-dimensional size of the volume from which to generate a
            first position to start the chain.

        :return sampler: array
            `emcee.EnsembleSampler` object that is used for posterior analysis
        """
        ndim = 5 + 2 * self.n_datasets
        pos = np.array([self.guess + ballsize * np.random.randn(ndim)
                        for i in range(nwalkers)])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                        threads=nthreads)
        sampler.run_mcmc(pos, nsteps)
        return sampler


# The following is used for testing when estimate.py is run by itself
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import corner

    # The "true" parameters
    k_true = 58.1E-3
    period_true = 2.98565
    t0_true = 1497.5
    w_true = 11.
    e_true = 0.213

    # Proper motions for different datasets and number of points to compute a
    # period of RVs
    vz = 29.027
    nt = 1000
    npoints = 100

    ts = np.linspace(1494., 1500., npoints)
    print('\nCreating mock data of radial velocities of HD83443 b.')
    HD83443 = orbit.BinarySystem(log_k=np.log10(k_true),
                                 log_period=np.log10(period_true),
                                 t0=t0_true,
                                 w=w_true,
                                 log_e=np.log10(e_true))
    rvs = HD83443.get_rvs(ts=ts, nt=nt)

    # "Observing" the data
    rv_d = np.array(
        [rvk + np.random.normal(loc=0., scale=0.015) for rvk in rvs])
    t_d = np.array([tk + np.random.normal(loc=0., scale=0.1) for tk in ts])
    rv_derr = np.array([0.015 + np.random.normal(loc=0.0, scale=0.005)
                        for k in rvs])

    # Adding an offset to the RVs
    rv_d += vz

    # We use the true values as the initial guess for the orbital parameters
    # The last two values are for the estimates of logf, which is the log10(f),
    # where f is the underestimating factor of the errorbars of each dataset
    _guess = [np.log10(k_true), np.log10(period_true), t0_true, w_true,
              np.log10(e_true), vz, 0.0]

    print('\n-------------------------------------------------------------')
    print('Starting maximum likelihood estimation.')
    start_time = time.time()

    # We instantiate the class OrbitalParams with our data
    estim = OrbitalParams(t_d, rv_d, rv_derr, guess=_guess, n_datasets=1,
                          bounds_vz=(25, 35),
                          bounds_sj=(-1, 1))

    # And run the estimation
    params_ml = estim.ml_orbit(disp=True, maxiter=20000)
    print('Orbital parameters estimation took %.4f seconds.' %
          (time.time() - start_time))
    print('\nResults:')
    print('K = %.3f, T = %.2f, t0 = %.1f, w = %.1f, e = %.3f, vz0 = %.3f, '
          'sj0 = %.3f' % (10 ** params_ml[0], 10 ** params_ml[1], params_ml[2],
                          params_ml[3], 10 ** params_ml[4], params_ml[5],
                          params_ml[6]))
    print('\n"True" values:')
    print('K = %.3f, T = %.2f, t0 = %.1f, w = %.1f, e = %.3f, vz0 = %.3f' %
          (k_true, period_true, t0_true, w_true, e_true, vz))

    print('\nFinished testing maximum likelihood estimation.')
    print('---------------------------------------------------------------')
    print('Starting emcee estimation. It can take a few minutes.')
    start_time = time.time()
    _sampler = estim.emcee_orbit(nwalkers=14,
                                 nsteps=5000,
                                 nthreads=4)
    _ncut = 1000
    _ndim = 7
    _samples = samples = _sampler.chain[:, _ncut:, :].reshape((-1, _ndim))
    print('\nOrbital parameters estimation took %.4f seconds.' %
          (time.time() - start_time))
    _samples[:, 6] = np.abs(_samples[:, 6])
    # corner is used to make these funky triangle plots
    print('Now creating the corner plot.')
    corner.corner(_samples,
                  labels=[r'$\log{K}$', r'$\log{T}$', r'$t_0$', r'$\omega$',
                          r'$\log{e}$', r'$v_{Z0}$', r'$s_{j0}$'],
                  truths=[np.log10(k_true), np.log10(period_true), t0_true,
                          w_true, np.log10(e_true), vz, 0.0])
    plt.savefig('corner.png')
    plt.show()

    # log to linear for some parameters
    _samples[:, 0] = 10 ** _samples[:, 0]
    _samples[:, 1] = 10 ** _samples[:, 1]
    _samples[:, 4] = 10 ** _samples[:, 4]
    # _samples[:, 7] = 10 ** _samples[:, 7]
    # _samples[:, 8] = 10 ** _samples[:, 8]

    # Printing results
    k_mcmc, period_mcmc, t0_mcmc, w_mcmc, e_mcmc, vz0_mcmc, f0_mcmc \
        = map(lambda v: np.array([v[1], v[2] - v[1], v[1] - v[0]]),
              zip(*np.percentile(_samples, [16, 50, 84], axis=0)))

    print('\nResults:')
    print('K = %.3f + (+ %.3f, -%.3f)' % (k_mcmc[0], k_mcmc[1], k_mcmc[2]))
    print('T = %.2f + (+ %.2f, -%.2f)' % (period_mcmc[0], period_mcmc[1],
                                          period_mcmc[2]))
    print('t0 = %.1f + (+ %.1f, -%.1f)' % (t0_mcmc[0], t0_mcmc[1], t0_mcmc[2]))
    print('w = %.1f + (+ %.1f, -%.1f)' % (w_mcmc[0], w_mcmc[1], w_mcmc[2]))
    print('e = %.3f + (+ %.3f, -%.3f)' % (e_mcmc[0], e_mcmc[1], e_mcmc[2]))
    print(
       'vz0 = %.3f + (+ %.3f, -%.3f)' % (vz0_mcmc[0], vz0_mcmc[1], vz0_mcmc[2]))
    print('sj0 = %.3f + (+ %.3f, -%.3f)' % (f0_mcmc[0], f0_mcmc[1], f0_mcmc[2]))
    print('\nFinished testing emcee estimation.')
    print('---------------------------------------------------------------')
"""
    print('Plotting the results.')

    # The results from MLE
    est_ml = orbit.BinarySystem(log_k=params_ml[0],
                                log_period=params_ml[1],
                                t0=params_ml[2],
                                w=params_ml[3],
                                log_e=params_ml[4])
    rvs_ml = est_ml.get_rvs(ts=ts, nt=nt)
    plt.plot(ts, rvs_ml, label='MLE')

    # The results from emcee
    est_mcmc = orbit.BinarySystem(log_k=np.log10(k_mcmc[0]),
                                  log_period=np.log10(period_mcmc[0]),
                                  t0=t0_mcmc[0],
                                  w=w_mcmc[0],
                                  log_e=np.log10(e_mcmc[0]))
    rvs_mcmc = est_mcmc.get_rvs(ts=ts, nt=nt)
    plt.plot(ts, rvs_mcmc, label='emcee')

    # Plotting various samples from MCMC
    s_redux = _samples[:, 0:-4]
    for k, T, t0, w, e in s_redux[np.random.randint(len(s_redux), size=200)]:
        est = orbit.BinarySystem(log_k=np.log10(k),
                                 log_period=np.log10(T),
                                 t0=t0,
                                 w=w,
                                 log_e=np.log10(e))
        rvs_sample = est.get_rvs(ts=ts, nt=nt)
        plt.plot(ts, rvs_sample, color="k", alpha=0.05)

    # The data
    for k in range(2):
        plt.errorbar(t_d[k], rv_d[k] - params_ml[5 + k], yerr=rv_derr[k],
                     fmt='.')
    plt.plot(ts, rvs, label='True orbit')

    plt.legend()
    plt.show()
"""