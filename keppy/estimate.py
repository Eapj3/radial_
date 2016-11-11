#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import scipy.optimize as op
from keppy import orbit, dataset
# import emcee
# import matplotlib
import matplotlib.pyplot as plt
# import corner
import lmfit

# Adjusting some useful matplotlib parameters
# matplotlib.rcParams.update({'font.size': 20})
# matplotlib.rc('xtick', labelsize=13)
# matplotlib.rc('ytick', labelsize=13)

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
    orbital period. IMPORTANT: All logarithms are log10, and not ln.

    Parameters
    ----------
    datasets : sequence or ``keppy.dataset.RVDataSet``
        A list of ``RVDataSet`` objects or one ``RVDataSet`` object that
        contains the data to be fit. If a sequence is passed, the order that
        the data sets in the sequence will dictate which instrumental parameter
        (gamma, sigma) index correspond to each data set.

    guess : ``dict`` or sequence
        First guess of the orbital parameters in the following order. If passed
        as a sequence, its length must be the same as the number of parameters
        of the fit: 5 orbital parameters + N, if use_add_sigma is ``False``; or
        5 orbital parameters + 2 * N, if use_add_sigma is ``True``, where N is
        the number of data sets. If passed as ``dict``, they keywords must be
        matched to the names of the parameters to be fit. These names are:
        'log_k', 'log_period', 't0', 'omega', 'log_e', 'sqe_cosw', 'sqe_sinw',
        'gamma', 'sigma', 'gamma_X', 'sigma_X', where 'X' is the index of the
        data set; 'omega' and 'log_e' are used in the 'mc10' parametrization;
        'sqe_cosw' and 'sqe_sinw' are used in the 'exofast' parametrization. If
        passed as ``dict`` and parameters are missing, the code uses the
        following default values: log_k=-1, log_period=3, t0=5000, omega=180,
        log_e=-1, sqe_cosw=0, sqe_sinw=0, gamma=0, sigma=0.001.

    bounds : ``dict``, optional
        Bounds of the parameter search, passed as a ``tuple`` for each
        parameter. The ``dict`` keywords must match the names of the parameters.
        These names are: 'log_k', 'log_period', 't0', 'omega', 'log_e',
        'sqe_cosw', 'sqe_sinw', 'gamma', 'sigma', 'gamma_X', 'sigma_X', where
        'X' is the index of the data set; 'omega' and 'log_e' are used in the
        'mc10' parametrization; 'sqe_cosw' and 'sqe_sinw' are used in the
        'exofast' parametrization. If parameters are missing, the code uses the
        following default values: log_e=(-3, 3), log_period=(-3, 5),
        t0=(0, 10000), omega=(0, 360), log_e=(-4, -0.0001), sqe_cosw=(-1, 1),
        sqe_sinw=(-1, 1), gamma=(-10, 10), sigma=(0.0001, 1.0).

    parametrization: ``str``, optional
        The options are: 'mc10' for the parametrization of Murray & Correia
        2010, and 'exofast' for the parametrization of Eastman et al. 2013.
        Default is 'mc10'.

    use_add_sigma : ``bool``, optional
        If ``True``, the code will use additional parameter to estimate an extra
        uncertainty term for each RV data set. Default is ``False``.
    """
    def __init__(self, datasets, guess, bounds=None, parametrization=None,
                 use_add_sigma=False):

        self.datasets = datasets

        if isinstance(datasets, dataset.RVDataSet):
            self.n_ds = 1
        else:
            self.n_ds = len(datasets)
            # Check if the datasets are passed as RVDataSet objects
            for dsk in self.datasets:
                assert isinstance(dsk,
                                  dataset.RVDataSet), 'The datasets must be ' \
                                                      'passed as RVDataSet ' \
                                                      'objects.'

        # Read the data
        self.t = []
        self.rv = []
        self.rv_unc = []
        self.meta = []
        for dsk in self.datasets:
            self.t.append(dsk.t)
            self.rv.append(dsk.rv)
            self.rv_unc.append(dsk.rv_unc)
            self.meta.append(dsk.table.meta)

        self.use_add_sigma = use_add_sigma

        # Setting the parametrization option
        if parametrization is None:
            self.parametrization = 'mc10'
        else:
            self.parametrization = parametrization

        # Setting the parameter keywords and the bounds
        self.keys = ['log_k', 'log_period', 't0']
        if self.parametrization == 'mc10':
            self.keys.append('omega')
            self.keys.append('log_e')
        elif self.parametrization == 'exofast':
            self.keys.append('sqe_cosw')
            self.keys.append('sqe_sinw')
        if self.n_ds == 1:
            self.keys.append('gamma')
            if self.use_add_sigma is True:
                self.keys.append('sigma')
        else:
            for i in range(self.n_ds):
                self.keys.append('gamma_{}'.format(i))
            if self.use_add_sigma is True:
                for i in range(self.n_ds):
                    self.keys.append('sigma_{}'.format(i))

        # The guess dict
        self.guess = {}
        if isinstance(guess, dict) is False:
            for i in range(len(self.keys)):
                self.guess[self.keys[i]] = guess[i]
        elif isinstance(guess, dict) is True:
            for key in self.keys:
                try:
                    self.guess[key] = guess[key]
                except KeyError:
                    self.guess[key] = None

        # Setting the orbital parameter bounds
        self.bounds = {}
        for key in self.keys:
            self.bounds[key] = None
        if bounds is not None:
            for key in self.keys:
                try:
                    self.bounds[key] = bounds[key]
                except KeyError:
                    pass
        else:
            pass

        # Initializing useful global variables
        self.lmfit_result = None

    # The RV model from Murray & Correia 2010
    @staticmethod
    def rv_model_mc10(t, log_k, log_period, t0, omega, log_e, gamma):
        """
        The radial velocities model from Murray & Correia 2010.

        Parameters
        ----------
        t : ``astropy.units.Quantity``
            Time.

        log_k : scalar
            Logarithm (base 10) of the radial velocity semi-amplitude.

        log_period : scalar
            Logarithm (base 10) of the orbital period.

        t0 : ``astropy.units.Quantity``
            Time of pariastron passage (time unit).

        omega : ``astropy.units.Quantity``
            Argument of periapse (angle unit).

        log_e : scalar
            Eccentricity of the orbit.

        gamma : ``astropy.units.Quantity``
            Proper motion of the barycenter (velocity unit).

        Returns
        -------
        rvs : ``astropy.units.Quantity``
            Radial velocity.
        """
        k = 10 ** log_k
        period = 10 ** log_period
        ecc = 10 ** log_e
        system = orbit.BinarySystem(k, period, t0, omega, ecc, gamma=gamma)
        rvs = system.get_rvs(t)
        return rvs

    # The RV model from EXOFAST
    @staticmethod
    def rv_model_exofast(t, log_k, log_period, t0, sqe_cosw, sqe_sinw, gamma):
        """
        The radial velocities model from EXOFAST (Eastman et al. 2013).

        Parameters
        ----------
        t : ``astropy.units.Quantity``
            Time.

        log_k : scalar
            Logarithm (base 10) of the radial velocity semi-amplitude.

        log_period : scalar
            Logarithm (base 10) of the orbital period.

        t0 : ``astropy.units.Quantity``
            Time of pariastron passage (time unit).

        sqe_cosw : scalar
            sqrt(ecc) * cos(omega).

        sqe_sinw : scalar
            sqrt(ecc) * sin(omega).

        gamma : ``astropy.units.Quantity``
            Proper motion of the barycenter (velocity unit).

        Returns
        -------
        rvs : ``astropy.units.Quantity``
            Radial velocity.
        """
        k = 10 ** log_k
        period = 10 ** log_period
        system = orbit.BinarySystem(k, period, t0, sqe_cosw=sqe_cosw,
                                    sqe_sinw=sqe_sinw, gamma=gamma)
        rvs = system.get_rvs(t)
        return rvs

    # The log-likelihood
    def lnlike(self, theta):
        """

        Parameters
        ----------
        theta

        Returns
        -------

        """
        v = theta.valuesdict()
        sum_res = 0
        for i in range(self.n_ds):

            # Compute the RVs using the appropriate model
            if self.parametrization == 'mc10':
                rvs = self.rv_model_mc10(self.t[i], v[self.keys[0]],
                                         v[self.keys[1]], v[self.keys[2]],
                                         v[self.keys[3]], v[self.keys[4]],
                                         v[self.keys[5 + i]])
            elif self.parametrization == 'exofast':
                rvs = self.rv_model_exofast(self.t[i], v[self.keys[0]],
                                            v[self.keys[1]], v[self.keys[2]],
                                            v[self.keys[3]], v[self.keys[4]],
                                            v[self.keys[5 + i]])

            # If user wants to estimate additional sigma
            if self.use_add_sigma is False:
                inv_sigma2 = (1. / (self.rv_unc[i] ** 2)).value
            elif self.use_add_sigma is True:
                log_sigma_j = np.log10(theta[self.keys[5 +
                                                       self.n_ds + i]])
                inv_sigma2 = (1. / (self.rv_unc[i] ** 2 + (10 ** log_sigma_j)
                                   ** 2)).value

            # The log-likelihood
            sum_res += np.sum((self.rv[i] - rvs).value ** 2 * inv_sigma2 +
                              np.log(2. * np.pi / inv_sigma2))
        return sum_res

    # Estimation using lmfit
    def lmfit_orbit(self, fix_param=None):
        """

        Parameters
        ----------
        fix_param : ``dict``

        Returns
        -------
        result : ``lmfit.MinimizerResult``

        """

        vary = {}
        for key in self.keys:
            vary[key] = True

        if fix_param is not None:
            for key in self.keys:
                try:
                    vary[key] = fix_param[key]
                except KeyError:
                    pass

        # The default bounds and guess
        default_bounds = {'log_k': (-4, 3), 'log_period': (-4, 5),
                          't0': (0, 10000), 'omega': (0, 360),
                          'log_e': (-4, -0.0001), 'sqe_cosw': (-1, 1),
                          'sqe_sinw': (-1, 1), 'gamma': (-10, 10),
                          'sigma': (1E-4, 5E-1)}
        default_guess = {'log_k': -1, 'log_period': 3, 't0': 5000, 'omega': 180,
                         'log_e': -1, 'sqe_cosw': 0,  'sqe_sinw': 0, 'gamma': 0,
                         'sigma': 0.001}
        for i in range(self.n_ds):
            default_bounds['gamma_{}'.format(i)] = default_bounds['gamma']
            default_bounds['sigma_{}'.format(i)] = default_bounds['sigma']
            default_guess['gamma_{}'.format(i)] = default_guess['gamma']
            default_guess['sigma_{}'.format(i)] = default_guess['sigma']

        params = lmfit.Parameters()

        for key in self.keys:
            if self.bounds[key] is None:
                self.bounds[key] = default_bounds[key]
            if self.guess[key] is None:
                self.guess[key] = default_guess[key]

            params.add(key, self.guess[key], vary=vary[key],
                       min=self.bounds[key][0], max=self.bounds[key][1])

        # Perform minimization
        self.lmfit_result = lmfit.minimize(self.lnlike, params, method='Nelder')
        lmfit.printfuncs.report_fit(self.lmfit_result.params, min_correl=0.5)

    # Plot lmfit_orbit result and residuals
    def plot_lmfit_result(self, fold=False):

        p = self.lmfit_result.params.valuesdict()
        if self.n_datasets == 1:
            t = np.linspace(min(self.t) - 100, max(self.t) + 100, 1000)
            rvs = self.rv_model_mc10(self.t, p['log_k'], p['log_period'],
                                     p['t0'], p['omega'], p['log_e'], 0)
            rv_curve = self.rv_model_mc10(t, p['log_k'], p['log_period'],
                                     p['t0'], p['omega'], p['log_e'], 0)
            resi = self.rv - rvs
            gamma = p['gamma']
            plt.errorbar(self.t, np.array(self.rv) - gamma, self.rv_err, fmt='o',
                         label='Data')
            plt.plot(t, rv_curve, label='Fit')
            plt.show()


    # Flat priors
    def flat(self, theta):
        """
        Computes a flat prior probability for a given set of parameters theta.

        Parameters
        ----------
        theta : ``dict``
            The orbital and instrumental parameters.

        Returns
        -------
        prob : ``float``
            The prior probability for a given set of orbital and instrumental
            parameters.
        """
        # TODO: Implement usage of MinimizerResult as an alternative
        assert isinstance(theta, lmfit.minimizer.MinimizerResult) is False, \
            'The use of MinimizerResult class is not implemented yet.'

        # Compute the eccentricity beforehand to impose a prior of e < 1 on it
        try:
            ecc = 10 ** theta['log_e']
        except KeyError:
            ecc = theta['sqe_cosw'] ** 2 + theta['sqe_sinw'] ** 2

        check = [self.bounds[key][0] < theta[key] < self.bounds[key][1]
                 for key in self.keys]
        if all(check) is True and ecc < 1:
            prob = 0.0
        else:
            prob = -np.inf
        return prob
'''
    # The probability
    def lnprob(self, theta):
        """
        This function calculates the ln of the probabilities to be used in the
        MCMC estimation.

        Parameters
        ----------
        theta: ``dict``
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
    def __init__(self, t, rv, rv_err, n_datasets=1, log_k_len=5, log_p_len=5):
        assert (isinstance(n_datasets, int) and n_datasets > 0), 'n_datasets ' \
            'must be an integer larger than 0.'

        self.G = 1.328E11   # Gravitational constant in unit involving M_Sun, km
        # and s
        self.t = t
        self.rv = rv
        self.rv_err = rv_err
        self.n_datasets = n_datasets
        self.log_k_len = log_k_len
        self.log_p_len = log_p_len

        # Initilizing useful global parameters
        self.best_values = []
        self.chisq = []
        self.log_k_list = None
        self.log_p_list = None

        """
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
        """

    # The RV model; for now, it only works with one dataset
    @staticmethod
    def rv_model(x, log_k, log_period, t0, omega, log_e, gamma):
        """

        Parameters
        ----------
        x
        log_k
        log_period
        t0
        omega
        log_e
        gamma

        Returns
        -------

        """
        system = orbit.BinarySystem(log_k, log_period, t0, omega, log_e,
                                    vz=gamma)
        rvs = system.get_rvs(x)
        return rvs

    # Compute the grid search
    def grid_search(self, ranges, fix_t0=-500000, verbose=False):

        # Compute the grids of K and T
        log_k_grid = np.linspace(ranges[0, 0], ranges[0, 1], self.log_k_len)
        log_p_grid = np.linspace(ranges[1, 0], ranges[1, 1], self.log_p_len)
        self.log_k_list = log_k_grid
        self.log_p_list = log_p_grid

        # Setup the model
        model = lmfit.Model(self.rv_model)

        # Set a fixed value for t0
        #model.set_param_hint('t0', value=fix_t0, vary=False)

        # Set the minima and maxima for omega and log_e
        model.set_param_hint('omega', min=0, max=360)
        model.set_param_hint('log_e', min=-4, max=-0.0001)

        # The first guess
        current = {'t0': 0., 'omega': 180., 'log_e': -0.1, 'gamma': 0.0}
        self.best_values = {'log_k': [], 'log_period': [], 't0': [],
                            'omega': [], 'log_e': []}
        self.chisq = []

        # For each fixed pair log_period and log_k, fit sqe_cosw, sqe_sinw and
        # gamma
        for lpk in log_p_grid:
            for lkk in log_k_grid:

                # Fix log_K and log_T
                model.set_param_hint('log_k', value=lkk, vary=False)
                model.set_param_hint('log_period', value=lpk, vary=False)

                # Set the parameters
                pars = model.make_params(log_k=lkk, log_period=lpk,
                                         t0=current['t0'],
                                         omega=current['omega'],
                                         log_e=current['log_e'],
                                         gamma=current['gamma'])
                # Perform the fit
                result = model.fit(self.rv, pars, x=self.t)

                # Print fit report
                if verbose is True:
                    print(result.fit_report())

                # Update the guess with the current best values
                current = result.best_values

                # If the current lkk is the first on the grid, save the
                # best_values to be used in the next lpk to save estimation time
                if lkk == log_k_grid[0]:
                    next_lpk = result.best_values

                # Save the fit parameters
                self.best_values['log_k'].append(current['log_k'])
                self.best_values['log_period'].append(current['log_period'])
                self.best_values['t0'].append(current['t0'])
                self.best_values['omega'].append(current['omega'])
                self.best_values['log_e'].append(current['log_e'])
                self.chisq.append(np.log10(result.chisqr))

            # Set current to next_lpk when going to the next lpk
            current = next_lpk

        np.save('62039_params.npy', np.array(self.best_values))
        np.save('62039_chisqr.npy', np.array(self.chisq))

    # Plot the chi-square maep
    def plot_map(self, chisq_filename=None, params_filename=None, ranges=None):

        if chisq_filename is not None:
            chisq_array = np.load(chisq_filename)
            self.chisq = chisq_array

        #if params_filename is not None:
        #    params_array = np.load(params_filename)
        #    self.best_values = params_array

        log_k_grid = np.linspace(ranges[0, 0], ranges[0, 1], self.log_k_len)
        log_p_grid = np.linspace(ranges[1, 0], ranges[1, 1], self.log_p_len)
        self.log_k_list = log_k_grid
        self.log_p_list = log_p_grid

        #log_k = np.array(self.best_values['log_k'])
        #log_period = np.array(self.best_values['log_period'])
        #t0 = np.array(self.best_values['t0'])
        #omega = np.array(self.best_values['omega'])
        #log_e = np.array(self.best_values['log_e'])

        self.chisq = np.reshape(self.chisq, [self.log_p_len, self.log_k_len])
        plt.contourf(self.log_p_list, self.log_k_list, self.chisq.T, cmap='bone', origin='lower')
        plt.colorbar()
        plt.show()
'''