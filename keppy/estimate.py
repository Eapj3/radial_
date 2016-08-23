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

    :param bounds_vz: tuple, optional
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
    def __init__(self, t, rv, rv_err, guess, bounds_vz=((-10, 10), (-10, 10)),
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
            self.bounds = bounds + (bounds_vz,)
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
        for i in range(self.n_datasets):
            nt = len(self.t[i])
            system = orbit.BinarySystem(log_k=theta[0], log_period=theta[1],
                                        t0=theta[2], w=theta[3], log_e=theta[4],
                                        vz=theta[5 + i])
            model = system.get_rvs(ts=self.t[i], nt=nt)
            inv_sigma2 = 1. / self.rv_err[i] ** 2
            sum_like += np.sum((self.rv[i] - model) ** 2 * inv_sigma2 +
                               np.log(2. * np.pi / inv_sigma2))
            sum_like *= -0.5
        return sum_like

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
                             options={'maxiter': maxiter, "disp": True})
        return result["x"]

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
        pos = np.array([self.guess + 1e-2 * np.random.randn(ndim)
                        for i in range(nwalkers)])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                        threads=nthreads)
        sampler.run_mcmc(pos, nsteps)

        fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(6, 6))
        axes[0, 0].plot(sampler.chain[:, :, 0].T)
        #axes[0].axhline(y=np.log10(k_true), linewidth=3, color='orange')
        axes[0, 0].set_ylabel(r'$K$')
        axes[1, 0].plot(sampler.chain[:, :, 1].T)
        #axes[1].axhline(y=np.log10(period_true), linewidth=3,
        #                color='orange')
        axes[1, 0].set_ylabel(r'$T$')
        axes[2, 0].plot(sampler.chain[:, :, 2].T)
        #axes[2].axhline(y=t0_true, linewidth=3, color='orange')
        axes[2, 0].set_ylabel(r'$t_0$')
        axes[0, 1].plot(sampler.chain[:, :, 3].T)
        # axes[3].axhline(y=w_true, linewidth=3, color='orange')
        axes[0, 1].set_ylabel(r'$w$')
        axes[1, 1].plot(sampler.chain[:, :, 4].T)
        # axes[4].axhline(y=np.log10(e_true), linewidth=3, color='orange')
        axes[1, 1].set_ylabel(r'$e$')
        axes[2, 1].plot(sampler.chain[:, :, 5].T)
        # axes[5].axhline(y=vz, linewidth=3, color='orange')
        axes[2, 1].set_ylabel(r'$V_Z$')
        plt.xlabel('step number')
        plt.show()

        samples = sampler.chain[:, :, :].reshape((-1, ndim))
        print(self.lnprob(samples[-1]))
        return samples, sampler


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
    vz = [29.027, 35.0]
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

    # Breaking the RV data in the middle in order to simulate different datasets
    new_rv_d = np.array([rv_d[:npoints//2].tolist(),
                         rv_d[npoints//2:].tolist()])
    new_t_d = np.array([t_d[:npoints//2].tolist(), t_d[npoints//2:].tolist()])
    new_rv_d[0] += vz[0]
    new_rv_d[1] += vz[1]
    rv_d = new_rv_d
    t_d = new_t_d
    rv_m = [np.mean(rv_d[0]), np.mean(rv_d[1])]
    print('Mean of RVs = %.3f, %.3f' % (rv_m[0], rv_m[1]))

    # Subtracting the mean of each RV dataset
    for i in range(len(rv_d)):
        rv_d[i] -= rv_m[i]

    # We use the true values as the initial guess for the orbital parameters
    _guess = [np.log10(k_true), np.log10(period_true), t0_true, w_true,
              np.log10(e_true), 0.0, 0.0]

    print('\n-------------------------------------------------------------')
    print('Starting maximum likelihood estimation.')
    start_time = time.time()

    # We instantiate the class OrbitalParams with our data
    estim = OrbitalParams(t_d, rv_d, rv_derr, guess=_guess,
                          bounds=((-2, -1), (0.4, 0.5), (1490, 1510), (5, 20),
                                  (-0.7, -0.6)),
                          bounds_vz=((-1, 1), (-1, 1)),
                          n_datasets=2)

    # And run the estimation
    params_ml = estim.ml_orbit()
    print('Orbital parameters estimation took %.4f seconds.' %
          (time.time() - start_time))
    print('\nResults:')
    print('K = %.3f, T = %.2f, t0 = %.1f, w = %.1f, e = %.3f, vz0 = %.3f, '
          'vz1 = %.3f' %
          (10 ** params_ml[0], 10 ** params_ml[1], params_ml[2],
           params_ml[3], 10 ** params_ml[4], params_ml[5], params_ml[6]))
    print('\n"True" values:')
    print('K = %.3f, T = %.2f, t0 = %.1f, w = %.1f, e = %.3f, vz0 = %.3f, '
          'vz1 = %.3f' %
          (k_true, period_true, t0_true, w_true, e_true, vz[0]-rv_m[0],
           vz[1]-rv_m[1]))

    # Plotting the results
    est = orbit.BinarySystem(log_k=params_ml[0],
                             log_period=params_ml[1],
                             t0=params_ml[2],
                             w=params_ml[3],
                             log_e=params_ml[4])
    rvs = est.get_rvs(ts=ts, nt=nt)
    for i in range(2):
        plt.errorbar(t_d[i], rv_d[i]-params_ml[5+i], yerr=rv_derr[i], fmt='.')
    plt.plot(ts, rvs)
    plt.show()
    print('\nFinished testing maximum likelihood estimation.')
    print('---------------------------------------------------------------')
    """
    print('Starting emcee estimation. It can take a few minutes.')
    estim = OrbitalParams(t_d, rv_d, rv_derr, guess=params_ml,
                          #bounds=((-3, -1), (0, 1), (1490, 1500), (0, 20),
                          #        (-3, -1)),
                          bounds_vz=(25, 30), dbglvl=1)
    start_time = time.time()
    _samples, sampler = estim.emcee_orbit(nwalkers=20,
                                 ncut=100,
                                 nsteps=1000,
                                 nthreads=4)
    print('\nOrbital parameters estimation took %.4f seconds.' %
          (time.time() - start_time))
    # corner is used to make these funky triangle plots
    print('Now creating the corner plot.')
    #corner.corner(_samples,
    #              labels=[r'$\ln{K}$', r'$\ln{T}$', r'$t_0$', r'$\omega$',
    #                      r'$\ln{e}$', r'$v_Z$'],
    #              truths=[np.log10(k_true), np.log10(period_true), t0_true,
    #                      w_true, np.log10(e_true), vz])
    #plt.show()

    # log to linear for some parameters
    _samples[:, 0] = 10 ** _samples[:, 0]
    _samples[:, 1] = 10 ** _samples[:, 1]
    _samples[:, 4] = 10 ** _samples[:, 4]

    #fig, axes = plt.subplots(3, sharex=True, figsize=(6, 6))
    #axes[0].plot(sampler.chain[:, :, 0].T)
    #axes[0].axhline(y=np.log10(k_true), linewidth=3, color='orange')
    #axes[0].set_ylabel(r'$K$')
    #axes[1].plot(sampler.chain[:, :, 1].T)
    #axes[1].axhline(y=np.log10(period_true), linewidth=3, color='orange')
    #axes[1].set_ylabel(r'$T$')
    #axes[2].plot(sampler.chain[:, :, 2].T)
    #axes[2].axhline(y=t0_true, linewidth=3, color='orange')
    #axes[2].set_ylabel(r'$t_0$')
    #axes[3].plot(sampler.chain[:, :, 3].T)
    #axes[3].axhline(y=w_true, linewidth=3, color='orange')
    #axes[3].set_ylabel(r'$w$')
    #axes[4].plot(sampler.chain[:, :, 4].T)
    #axes[4].axhline(y=np.log10(e_true), linewidth=3, color='orange')
    #axes[4].set_ylabel(r'$e$')
    #axes[5].plot(sampler.chain[:, :, 5].T)
    #axes[5].axhline(y=vz, linewidth=3, color='orange')
    #axes[5].set_ylabel(r'$V_Z$')
    #plt.xlabel('step number')
    #plt.show()

    # Printing results
    k_mcmc, period_mcmc, t0_mcmc, w_mcmc, e_mcmc, vz_mcmc = map(
        lambda v: np.array([v[1], v[2] - v[1], v[1] - v[0]]),
        zip(*np.percentile(_samples, [16, 50, 84], axis=0)))

    print('\nResults:')
    print('K = %.3f + (+ %.3f, -%.3f)' % (k_mcmc[0], k_mcmc[1], k_mcmc[2]))
    print('T = %.2f + (+ %.2f, -%.2f)' % (period_mcmc[0], period_mcmc[1],
                                          period_mcmc[2]))
    print('t0 = %.1f + (+ %.1f, -%.1f)' % (t0_mcmc[0], t0_mcmc[1], t0_mcmc[2]))
    print('w = %.1f + (+ %.1f, -%.1f)' % (w_mcmc[0], w_mcmc[1], w_mcmc[2]))
    print('e = %.3f + (+ %.3f, -%.3f)' % (e_mcmc[0], e_mcmc[1], e_mcmc[2]))
    print('vz = %.3f + (+ %.3f, -%.3f)' % (vz_mcmc[0], vz_mcmc[1], vz_mcmc[2]))
    print('\nFinished testing emcee estimation')
    print('---------------------------------------------------------------')"""
