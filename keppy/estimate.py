#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as op
from keppy import orbit
import emcee
import matplotlib
import matplotlib.pyplot as plt
import corner

# Adjusting some useful matplotlib parameters
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('xtick', labelsize=13)
matplotlib.rc('ytick', labelsize=13)

"""
This code contains routines to estimate the orbital parameters of a binary
system by means of maximum likelihood estimation or a Markov-Chain Monte Carlo
estimation using the code emcee. Before using it, it is highly recommended to
read the documentation of emcee in order to understand what are priors,
probabilities, sampling and other jargon. Check it out at
http://dan.iel.fm/emcee/current/
"""


# Estimate orbital parameters from radial velocity data comprising at least one
# orbit.
class FullOrbit(object):
    """
    A class that computes the orbital parameters of a binary system given its
    radial velocities (and their uncertainties) in function of time. This class
    is optimized for timeseries that contain at least one full or almost full
    orbital period.

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

        # Initializing useful global variables
        self.sampler = None
        self.samples = None
        self.samples_EXOFAST = None
        self.labels_EXOFAST = None
        self.dyn_mcmc = None
        self.params_mcmc = None
        self.system_mcmc = None

        # Dealing with the parameter labels for plots
        labels_orbit = [r'$\log{K}$', r'$\log{T}$', r'$t_0$',
                        r'$\sqrt{e}\cos{\omega}$', r'$\sqrt{e}\sin{\omega}$']
        # If the user did not use extra-noise term, no labels for sigma.
        # Otherwise, include labels for the various sigma for each dataset.
        if self.bounds_sj is not None:
            labels_gamma = []
            labels_sigma = []
            for i in range(self.n_datasets):
                labels_gamma.append(r'$\gamma_{%s}$' % str(i))
                labels_sigma.append(r'$\sigma_{%s}$' % str(i))
            self.labels = labels_orbit + labels_gamma + labels_sigma
        else:
            labels_gamma = []
            for i in range(self.n_datasets):
                labels_gamma.append(r'$\gamma_{%s}$' % str(i))
            self.labels = labels_orbit + labels_gamma

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
    def emcee_orbit(self, nwalkers=20, nsteps=1000, p_scale=2.0, nthreads=1,
                    ballsizes=1E-2):
        """
        Calculates samples of parameters that best fit the signal rv.

        Parameters
        ----------
        nwalkers : ``int``
            Number of walkers

        nsteps : ``int``
            Number of burning-in steps

        p_scale : ``float``, optional
            The proposal scale parameter. Default is 2.0.

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
                                        a=p_scale, threads=nthreads)
        sampler.run_mcmc(pos, nsteps)
        self.sampler = sampler

    # Plot emcee chains
    def plot_emcee_chains(self, outfile='chains.pdf', n_cols=2,
                          fig_size=(12, 12)):
        """
        Plot the ``emcee`` chains so that the user can check for convergence or
        chain behavior.

        Parameters
        ----------
        outfile : ``str`` or ``None``, optional
            Name of the output image file to be saved. If ``None``, then no
            output file is produced, and the plot is displayed on screen.
            Default is 'chains.pdf'.

        n_cols : ``int``, optional
            Number of columns of the plot. Default is 2.

        fig_size : tuple, optional
            Sizes of each panel of the plot, where the first element of the
            tuple corresponds to the x-direction size, and the second element
            corresponds to the y-direction size. Default is (12, 12).
        """
        assert (self.sampler is not None), "The emcee sampler must be run " \
                                           "before plotting the chains."
        n_walkers, n_steps, n_params = np.shape(self.sampler.chain)

        # Dealing with the number of rows for the plot
        if n_params % n_cols > 0:
            n_rows = n_params // n_cols + 1
        else:
            n_rows = n_params // n_cols

        # Finally Do the actual plot
        ind = 0     # The parameter index
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, sharex=True,
                                 figsize=fig_size)
        for i in range(n_cols):
            for k in range(n_rows):
                if ind < len(self.labels):
                    axes[k, i].plot(self.sampler.chain[:, :, ind].T)
                    axes[k, i].set_ylabel(self.labels[ind])
                    ind += 1
                else:
                    pass
            plt.xlabel('Step number')
        if outfile is None:
            plt.show()
        else:
            plt.savefig(outfile)

    # Compute samples from the emcee chains.
    def make_samples(self, n_cut, save_file=None):
        """
        Compute the MCMC samples from the ``emcee`` chains. The user has to
        provide the number of steps to ignore in the beginning of the chain
        (which correspond to the burn-in phase), compute eccentricities and
        arguments of periapse and saving the samples to a file on disk.

        Parameters
        ----------
        n_cut : ``int``
            Number of steps to ignore from the burn-in phase.

        save_file : ``str`` or ``None``, optional
            Output file to save the samples to a file on disk. If ``None``, no
            output file is produced. Default is ``None``. Note: These files
            are not compressed, and will be saved using ``numpy.save``; they can
            be loaded using ``numpy.load``.
        """
        assert (self.sampler is not None), "The emcee sampler must be run " \
                                           "before computing samples."

        n_walkers, n_steps, n_params = np.shape(self.sampler.chain)

        # Save samples by cutting the burn-in phase
        self.samples = self.sampler.chain[:, n_cut:, :].reshape((-1, n_params))

        # Compute the eccentricity (e) and the argument of periapse (omega) of
        # the orbit if the user requested so, and save them in place of
        # sqrt(e)*cos(omega) and sqrt(e)*sin(omega). The samples of the original
        # parameters will be kept as samples_EXOFAST.
        ecc = (self.samples[:, 3]) ** 2 + (self.samples[:, 4]) ** 2
        cosw = (self.samples[:, 3]) / np.sqrt(ecc)
        sinw = (self.samples[:, 4]) / np.sqrt(ecc)
        omega = np.degrees(np.arctan2(sinw, cosw))
        self.samples_EXOFAST = self.samples
        self.samples[:, 4] = ecc
        self.samples[:, 3] = omega
        # Also change the labels
        self.labels_EXOFAST = self.labels
        self.labels[4] = r'$e$'
        self.labels[3] = r'$\omega$'

        # Save samples to file if the user requested so
        if save_file is not None:
            np.save(file=save_file, arr=self.samples)

    # Make the corner plot for the samples
    def plot_corner(self, bins=20, out_file='corner.pdf'):
        """
        Make the corner plots.

        Parameters
        ----------
        bins: ``str``, optional
            Number of bins. Default is 20.

        out_file : ``str`` or ``None``, optional
            Name of the output image file. If ``None``, no outpuf file is
            produced and the plot is displayed on screen. Default is
            'corner.pdf'.
        """
        assert (self.samples is not None), "The samples must be computed " \
                                           "before making the corner plot."
        corner.corner(self.samples, bins, labels=self.labels)
        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)

    # Compute the companion minimum mass and the semi-major axis of the orbit.
    def compute_dynamics(self, main_body_mass=1.0):
        """
        Compute the mass and semi-major axis of the companion defined by the
        orbital parameters estimated with ``emcee``.

        Parameters
        ----------
        main_body_mass : ``float``, optional
            The mass of the main body which the companion orbits, in units of
            solar masses. Default is 1.0.
        """
        mbm = main_body_mass
        k = 10 ** self.samples[:, 0]
        period = 10 ** self.samples[:, 1] * 8.64E4
        log_2pi_grav = 11.92138   # Logarithm of 2 * np.pi * G in units of
        # km ** 3 * s ** (-2) * M_Sun ** (-1)
        # ``eta`` is the numerical value of the following equation
        # period * K * (1 - e ** 2) ** (3 / 2) / 2 * pi * G / main_body_mass
        log_eta = np.log10(period) + 3 * np.log10(k) + \
            3. / 2 * np.log10(1. - self.samples[:, 4] ** 2) - log_2pi_grav
        eta = 10 ** log_eta / mbm

        # Find the zeros of the third order polynomial that relates ``msini``
        # to ``eta``. The first zero is the physical ``msini``.
        roots = np.array([np.roots([1, -ek, -2 * ek, -ek]) for ek in eta])
        msini = abs(roots[:, 0])

        # Compute the semi-major axis in km and convert to AU
        semi_a = np.sqrt(1.328E11 / (2 * np.pi) * msini * period / k /
                         np.sqrt(1. - self.samples[:, 4] ** 2))
        semi_a *= 6.68458712E-9
        self.dyn_mcmc = np.array([msini, semi_a])
        return msini, semi_a

    # Print emcee results in an objective way
    # noinspection PyArgumentList
    def print_emcee_results(self):
        """

        :return:
        """
        linear_samples = np.zeros_like(self.samples)
        for i in range(len(self.samples[0, :])):
            linear_samples[:, i] = self.samples[:, i]
        linear_samples[:, 0] = 10 ** linear_samples[:, 0]
        linear_samples[:, 1] = 10 ** linear_samples[:, 1]
        labels = ['K', 'P', 't0', 'omega', 'ecc']
        gamma_labels = []
        sigma_labels = []

        units = ['km/s', 'days', 'JD-2.45E6 days', 'deg', '']
        dyn_units = ['M_Sun', 'AU']

        for i in range(self.n_datasets):
            gamma_labels.append('gamma_%i' % i)
        if self.bounds_sj is not None:
            for i in range(self.n_datasets):
                linear_samples[:, -1 - i] = 10 ** self.samples[:, -1 - i]
                sigma_labels.append('addsigma_%i' % i)

        k_mcmc, period_mcmc, t0_mcmc, ecc_mcmc, omega_mcmc = \
            map(lambda v: np.array([v[1], v[2]-v[1], v[1]-v[0]]),
                zip(*np.percentile(linear_samples[:, :5], [16, 50, 84],
                                   axis=0)))
        self.params_mcmc = [k_mcmc, period_mcmc, t0_mcmc, ecc_mcmc, omega_mcmc]

        msini_mcmc, semi_a_mcmc = \
            map(lambda v: np.array([v[1], v[2]-v[1], v[1]-v[0]]),
                zip(*np.percentile(self.dyn_mcmc.T, [16, 50, 84], axis=0)))
        self.system_mcmc = [msini_mcmc, semi_a_mcmc]

        # Print results
        for i in range(5):
            print('%s = %.3f ^{+ %.3f}_{- %.3f} %s' % (labels[i],
                                                       self.params_mcmc[i][0],
                                                       self.params_mcmc[i][1],
                                                       self.params_mcmc[i][2],
                                                       units[i]))
        print('msini = %.3f ^{+ %.3f}_{- %.3f} M_Sun' % (self.system_mcmc[0][0],
                                                         self.system_mcmc[0][1],
                                                         self.system_mcmc[0][2])
              )
        print('a = %.3f ^{+ %.3f}_{- %.3f} AU' % (self.system_mcmc[1][0],
                                                  self.system_mcmc[1][1],
                                                  self.system_mcmc[1][2]))

    # Plot emcee solutions
    def plot_folded_solutions(self, labels, fmts, n_samples=200, curve_len=1000,
                             outfile=None):
        """

        :param labels:
        :param fmts:
        :param n_samples:
        :param outfile:
        :return:
        """
        ts = np.linspace(0, 1, 1000)
        orbit_samples = self.samples[:, :5]

        # Plot a random sample of curves
        for _logK, _logT, _t0, _w, _e in \
                orbit_samples[np.random.randint(len(orbit_samples),
                                                size=n_samples)]:
            orbit_est = orbit.BinarySystem(log_k=_logK, log_period=_logT,
                                           t0=_t0, w=_w, log_e=np.log10(_e))
            rv_est = orbit_est.get_rvs(nt=curve_len, ts=ts * 10 ** _logT)
            plt.plot(ts, rv_est, color="k", alpha=0.02)

        # Plot the best result
        mcmc_est = orbit.BinarySystem(log_k=np.log10(self.params_mcmc[0][0]),
                                      log_period=np.log10(
                                          self.params_mcmc[1][0]),
                                      t0=self.params_mcmc[2][0],
                                      w=self.params_mcmc[3][0],
                                      log_e=np.log10(self.params_mcmc[4][0]))
        rv_mcmc = mcmc_est.get_rvs(nt=curve_len, ts=ts * self.params_mcmc[1][0])
        plt.plot(ts, rv_mcmc, label="MCMC", color='purple', lw=2)

        plt.show()


# Estimate orbital parameters from radial velocity data comprising only a linear
# trend.
class LinearTrend(object):
    """
    The method applied in this class is based on the partial orbits method used
    by Wright et al. 2007, ApJ 657.
    """
    def __init__(self, t, rv, rv_err, param_arrays, t0, m_star, n_datasets=1):
        assert (isinstance(n_datasets, int) and n_datasets > 0), 'n_datasets ' \
            'must be an integer larger than 0.'

        self.G = 1.328E11   # Gravitational constant in unit involving M_Sun, km
        # and s
        self.t = t
        self.rv = rv
        self.rv_err = rv_err
        self.n_datasets = n_datasets
        self.t0 = t0

        self.log_msini = param_arrays[0]
        self.log_period = param_arrays[1]
        self.e = param_arrays[2]
        self.omega = param_arrays[3]
        self.gamma = param_arrays[4]

        # Computing useful parameters
        self.m = 10 ** self.log_msini
        self.n = 2 * np.pi / 10 ** self.log_period
        self.log_e = np.log10(self.e)
        self.sqe_cosw = np.sqrt(self.e) * np.cos(self.omega)
        self.sqe_sinw = np.sqrt(self.e) * np.sin(self.omega)
        self.semi_a = ((self.m + m_star) * self.G / self.n ** 2) ** (1./3)
        self.log_k = np.log10(self.m / (m_star + self.m) * self.n *
                              self.semi_a / np.sqrt(1. - self.e ** 2))

    # Compute chi-squared of fit
    # noinspection PyTypeChecker
    def chi_sq(self, theta):

        chi_sq = 0
        for i in range(self.n_datasets):
            if self.n_datasets > 1:
                n = len(self.t[i])
            else:
                n = len(self.t[0])
            system = orbit.BinarySystem(log_k=theta[0], log_period=theta[1],
                                        t0=theta[2], sqe_cosw=theta[3],
                                        sqe_sinw=theta[4], vz=theta[5 + i])
            model = system.get_rvs(ts=self.t[i], nt=n)
            chi_sq += np.sum((self.rv[i] - model) ** 2 / model /
                             self.rv_err[i] ** 2)
        return chi_sq
