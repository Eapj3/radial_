#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from radial import orbit, dataset, rv_model, prior
import scipy.signal as ss
import matplotlib.pyplot as plt
import matplotlib.markers as mrk
import lmfit
import corner
import emcee
import astropy.units as u
import astropy.constants as c

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
    is optimized for time series that contain at least one full or almost full
    orbital period. IMPORTANT: All logarithms are log10, and not ln.

    Parameters
    ----------
    datasets : sequence or ``keppy.dataset.RVDataSet``
        A list of ``RVDataSet`` objects or one ``RVDataSet`` object that
        contains the data to be fit. If a sequence is passed, the order that
        the data sets in the sequence will dictate which instrumental parameter
        (gamma, sigma) index correspond to each data set.

    guess : ``dict``
        First guess of the orbital parameters. The keywords must match to the
        names of the parameters to be fit. These names are: ``'log_k'``,
        ``'log_period'``, ``'t0'``, ``'omega'``, ``'ecc'``, ``'sqe_cosw'``,
        ``'sqe_sinw'``, ``'gamma_X'``, ``'sigma_X'``, where 'X' is the index of
        the data set; ``'omega'`` and ``'log_ecc'`` are used in the ``'mc10'``
        parametrization; ``'sqe_cosw'`` and ``'sqe_sinw'`` are used in the
        ``'exofast'`` parametrization. If parameters are missing, uses the
        following default first guesses: k=100 m/s, period=1000 days, t0=5000
        days, omega=np.pi, ecc=0.1, sqe_cosw=0, sqe_sinw=0, gamma=0, sigma=1
        m/s.

    bounds : ``dict`` or ``None``, optional
        Bounds of the parameter search, passed as a ``tuple`` for each
        parameter. The ``dict`` keywords must match the names of the parameters.
        See the description of ``guess`` for a reference. If parameters are
        missing, the uses the following default values: log_k=(0, 6) dex(m/s),
        log_period=(-3, 5) dex(days), t0=(0, 10000) days, omega=(0, 2 * np.pi),
        log_ecc=(-4, -0.0001), sqe_cosw=(-1, 1), sqe_sinw=(-1, 1),
        gamma=(-10000, 10000) m/s, sigma=(0.1, 500.0) m/s.

    parametrization: ``str``, optional
        The options are: ``'mc10'`` for the parametrization of Murray & Correia
        2010, and ``'exofast'`` for the parametrization of Eastman et al. 2013.
        Default is ``'mc10'``.

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
            self.t.append(dsk.t.to(u.d).value)
            self.rv.append(dsk.rv.to(u.m / u.s).value)
            self.rv_unc.append(dsk.rv_unc.to(u.m / u.s).value)
            self.meta.append(dsk.table.meta)

        self.use_add_sigma = use_add_sigma

        # Setting the parametrization option
        if parametrization is None:
            self.parametrization = 'mc10'
        else:
            self.parametrization = parametrization

        # The default bounds and guess
        self.default_bounds = {'log_k': (-3, 3),
                               'log_period': (-3, 5),
                               't0': (0, 10000)}
        self.default_guess = {'log_k': 2,
                              'log_period': 3,
                              't0': 5000}
        if self.parametrization == 'mc10':
            self.default_bounds['omega'] = (0, 2 * np.pi)
            self.default_bounds['log_ecc'] = (-4, -0.0001)
            self.default_guess['omega'] = np.pi
            self.default_guess['log_ecc'] = -1
        elif self.parametrization == 'exofast':
            self.default_bounds['sqe_cosw'] = (-1, 1)
            self.default_bounds['sqe_sinw'] = (-1, 1)
            self.default_guess['sqe_cosw'] = 0
            self.default_guess['sqe_sinw'] = 0
        for i in range(self.n_ds):
            self.default_bounds['gamma_{}'.format(i)] = (-10000, 10000)
            self.default_guess['gamma_{}'.format(i)] = 0
            if self.use_add_sigma is True:
                self.default_bounds['sigma_{}'.format(i)] = (0.1, 500)
                self.default_guess['sigma_{}'.format(i)] = 1

        # The global parameter keywords to be used in the code
        self.keys = self.default_guess.keys()

        # Setting up the working guess and bounds dicts ########################
        self.guess = {}
        for key in self.keys:
            try:
                self.guess[key] = guess[key]
            except KeyError:
                self.guess[key] = self.default_guess[key]

        self.bounds = {}
        for key in self.keys:
            self.bounds[key] = self.default_bounds[key]
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
        self.residuals = None
        self.sampler = None
        self.best_params = {}
        for key in self.keys:
            self.best_params[key] = None

    # Compute a periodogram of a data set
    def lomb_scargle(self, dset_index, freqs):
        """
        Compute a Lomb-Scargle periodogram for a given data set using
        ``scipy.signal.lombscargle``.

        Parameters
        ----------
        dset_index : ``int``
            Index of the data set to have the periodogram calculated for.

        freqs : ``array_like``
            Angular frequencies for output periodogram.

        Returns
        -------
        pgram : ``array_like``
            Lomb-Scargle periodogram.

        fig : ``matplotlib.pyplot.figure``
            Periodogram plot.
        """
        x_array = self.t[dset_index]
        y_array = self.rv[dset_index]
        pgram = ss.lombscargle(x=x_array, y=y_array, freqs=freqs)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.semilogx(freqs, pgram)
        return pgram, fig

    # Plot the data sets
    def plot_ds(self, legend_loc=None, symbols=None, plot_guess=False,
                fold=False, numpoints=1000):
        """
        Plot the data sets.

        Parameters
        ----------
        legend_loc : ``int`` or ``None``, optional
            Location of the legend. If ``None``, use the default from
            ``matplotlib``. Default is ``None``.

        symbols : sequence or ``None``, optional
            List of symbols for each data set in the plot. If ``None``, use
            the default list from ``matplotlib`` markers. Default is ``None``.

        plot_guess : ``bool``, optional
            If ``True``, also plots the guess as a black curve, and the RVs of
            each data set is shifted by its respective gamma value.
        """
        # Use matplotlib's default symbols if ``None`` is passed.
        if symbols is None:
            markers = mrk.MarkerStyle()
            symbols = markers.filled_markers

        fig = plt.figure(figsize=(6, 5))
        gs = plt.GridSpec(2, 1, height_ratios=(4, 1))
        ax_fit = fig.add_subplot(gs[0])

        self.residuals = []
        for i in range(self.n_ds):
            if plot_guess is True:
                ax_res = fig.add_subplot(gs[1], sharex=ax_fit)

                # First we figure out the bounds of the plot
                t_min = min([min(tk) for tk in self.t])
                t_max = max([max(tk) for tk in self.t])
                if fold is False:
                    t_guess = np.linspace(t_min, t_max, numpoints)
                else:
                    t_guess = np.linspace(0, 1, numpoints)

                # Compute the radial velocities for the guess
                try:
                    system = orbit.BinarySystem(k=10 ** self.guess['log_k'],
                                                period=10 **
                                                       self.guess['log_period'],
                                                t0=self.guess['t0'],
                                                omega=self.guess['omega'],
                                                ecc=10 ** self.guess['log_ecc'],
                                                gamma=0)
                except KeyError:
                    system = orbit.BinarySystem(k=10 ** self.guess['log_k'],
                                                period=10 **
                                                       self.guess['log_period'],
                                                t0=self.guess['t0'],
                                                sqe_cosw=self.guess['sqe_cosw'],
                                                sqe_sinw=self.guess['sqe_sinw'],
                                                gamma=0)
                if fold is False:
                    rv_guess = system.get_rvs(ts=t_guess)
                else:
                    rv_guess = system.get_rvs(ts=t_guess * 10 **
                                              self.guess['log_period'])
                rv_guess_samepoints = system.get_rvs(ts=self.t[i])

                # Shift the radial velocities with the provided gamma
                rvs = self.rv[i] - self.guess['gamma_{}'.format(i)]

                # Compute residuals
                res = rv_guess_samepoints - rvs
                self.residuals.append(res)

                # And finally
                if fold is False:
                    # Plot the data and the curve
                    ax_fit.errorbar(self.t[i], rvs, yerr=self.rv_unc[i],
                                    fmt=symbols[i],
                                    label=self.meta[i]['Instrument'])
                    ax_fit.plot(t_guess, rv_guess, color='k')
                    # Plot the residuals
                    ax_res.errorbar(self.t[i], res, yerr=self.rv_unc[i],
                                    fmt=symbols[i])
                    ax_res.set_ylabel('Residuals\n(m / s)')
                    plt.setp(ax_res.get_xticklabels(), visible=False)
                else:
                    # Plot the data and the curve
                    phase = (self.t[i] / 10 ** self.guess['log_period']) % 1
                    ax_fit.errorbar(phase, rvs, yerr=self.rv_unc[i],
                                    fmt=symbols[i],
                                    label=self.meta[i]['Instrument'])
                    ax_fit.plot(t_guess, rv_guess, color='k')
                    # Plot the residuals
                    ax_res.errorbar(phase, res, yerr=self.rv_unc[i],
                                    fmt=symbols[i])
                    ax_res.set_ylabel('Residuals\n(m / s)')
                    plt.setp(ax_res.get_xticklabels(), visible=False)
            else:
                if fold is False:
                    ax_fit.errorbar(self.t[i], self.rv[i], yerr=self.rv_unc[i],
                                    fmt=symbols[i],
                                    label=self.meta[i]['Instrument'])
                else:
                    phase = (self.t[i] / 10 ** self.guess['log_period']) % 1
                    ax_fit.errorbar(phase, self.rv[i], yerr=self.rv_unc[i],
                                    fmt=symbols[i],
                                    label=self.meta[i]['Instrument'])

        # Show the plot
        if fold is False:
            ax_fit.set_xlabel('Time (d)')
        else:
            ax_fit.set_xlabel('Phase')
        ax_fit.set_ylabel('Radial velocities (m / s)')
        ax_fit.set_title('{}'.format(self.meta[0]['Target']))
        ax_fit.legend(loc=legend_loc, numpoints=1)
        plt.tight_layout()

        if plot_guess is False:
            return ax_fit
        else:
            return fig, gs

    # The log-likelihood
    def lnlike(self, theta):
        """
        Log-likelihood of a given set of parameters to adequately describe the
        observed data.

        Parameters
        ----------
        theta : ``dict``
            Parameter dictionary like to ``~estimate.FullOrbit.guess``.

        Returns
        -------
        sum_res : scalar
            The log-likelihood.
        """
        try:
            # If ``theta`` is an ``lmfit.Parameters`` object
            v = theta.valuesdict()
        except AttributeError:
            # If ``theta`` is a regular ``dict`` object
            v = theta

        sum_res = 0

        for i in range(self.n_ds):

            # Compute the RVs using the appropriate model
            if self.parametrization == 'mc10':
                rvs = rv_model.mc10(self.t[i], v['log_k'], v['log_period'],
                                    v['t0'], v['omega'], v['log_ecc'],
                                    v['gamma_{}'.format(i)])
            elif self.parametrization == 'exofast':
                rvs = rv_model.exofast(self.t[i], v['log_k'], v['log_period'],
                                    v['t0'], v['sqe_cosw'], v['sqe_sinw'],
                                    v['gamma_{}'.format(i)])

            # If user wants to estimate additional sigma
            if self.use_add_sigma is False:
                inv_sigma2 = (1. / (self.rv_unc[i] ** 2))
            elif self.use_add_sigma is True:
                log_sigma_j = np.log10(theta['sigma_{}'.format(i)])
                inv_sigma2 = (1. / (self.rv_unc[i] ** 2 +
                                    (10 ** log_sigma_j) ** 2))

            # The log-likelihood
            sum_res += np.sum((self.rv[i] - rvs) ** 2 * inv_sigma2 +
                              np.log(2. * np.pi / inv_sigma2))
        return sum_res

    # Prepare an ``lmfit.Parameters`` object
    def prepare_params(self, theta, bounds, vary_param=None):
        """
        Prepare a ``lmfit.Parameters`` object to be used in the ``lmfit``
        estimation.

        Parameters
        ----------
        theta : ``dict``
            The current orbital parameters.

        bounds : ``dict``
            The bounds of the search. Each key must contain a tuple whose first
            value is the minimum bound and the second is the maximum bound.

        vary_param : ``dict``
            Dictionary that says which parameters will vary in the estimation.
            By default, all parameters vary. A parameter can be fixed by setting
            its key to ``False``.

        Returns
        -------
        params : ``lmfit.Parameters``
            The ``lmfit.Parameters`` object for the estimation.
        """
        vary = {}
        for key in self.keys:
            vary[key] = True

        if vary_param is not None:
            for key in self.keys:
                try:
                    vary[key] = vary_param[key]
                except KeyError:
                    pass

        params = lmfit.Parameters()

        # Set the default values of bounds and guess for the missing parameters
        for key in self.keys:
            if bounds[key] is None:
                bounds[key] = self.default_bounds[key]
            if theta[key] is None:
                theta[key] = self.default_guess[key]

        # Generate the parameters object
        for key in self.keys:
            params.add(key, theta[key], vary=vary[key],
                       min=bounds[key][0], max=bounds[key][1])
        return params

    # Estimation using lmfit
    def lmfit_orbit(self, vary=None, verbose=True, update_guess=False,
                    minimize_mode='Nelder'):
        """
        Perform a fit to the radial velocities datasets using
        ``lmfit.minimize``.

        Parameters
        ----------
        vary : ``dict`` or ``None``, optional
            Dictionary with keywords corresponding to each parameter, and
            entries that are ``True`` if the parameter is to be left to vary, or
            ``False`` if the parameter is to be fixed in the value provided by
            the guess. If ``None``, all parameters will vary. Default is
            ``None``.

        verbose : ``bool``, optional
            If ``True``, print output in the screen. Default is ``False``.

        update_guess : ``bool``, optional
            If ``True``, updates ``~estimate.FullOrbit.guess`` with the
            estimated values from the minimization. Default is ``False``.

        minimize_mode : ``str``, optional
            The minimization algorithm string. See the documentation of
            ``lmfit.minimize`` for a list of options available. Default is
            ``'Nelder'``.

        Returns
        -------
        result : ``lmfit.MinimizerResult``
            The resulting ``MinimizerResult`` object.
        """
        # Prepare the ``lmfit.Parameters`` object
        params = self.prepare_params(self.guess, self.bounds, vary)

        # Perform minimization
        self.lmfit_result = lmfit.minimize(self.lnlike, params,
                                           method=minimize_mode)

        # Updating global variable best_params
        for key in self.keys:
            self.best_params[key] = self.lmfit_result.params[key].value

        if update_guess is True:
            self.guess = self.best_params

        if verbose is True:
            for key in self.keys:
                print('{} = {}'.format(key, self.best_params[key]))

        return self.best_params

    def compute_dynamics(self, main_body_mass=1.0):
        """
        Compute the mass and semi-major axis of the companion defined by the
        orbital parameters.

        Parameters
        ----------
        main_body_mass : ``float``, optional
            The mass of the main body which the companion orbits, in units of
            solar masses. Default is 1.0.
        """
        mbm = main_body_mass
        log_k = self.guess['log_k']
        log_period = self.guess['log_period']
        ecc = 10 ** self.guess['log_ecc']
        grav = c.G.to(u.km ** 3 / (u.solMass * u.d ** 2)).value
        log_2pi_grav = np.log10(2 * np.pi * grav)
        # Logarithm of 2 * np.pi * G in units of
        # km ** 3 * s ** (-2) * M_Sun ** (-1)
        # ``eta`` is the numerical value of the following equation
        # period * K * (1 - e ** 2) ** (3 / 2) / 2 * pi * G / main_body_mass
        log_eta = log_period + 3 * log_k + \
                  3. / 2 * np.log10(1. - ecc ** 2) - log_2pi_grav
        eta = 10 ** log_eta / mbm

        # Find the zeros of the third order polynomial that relates ``msini``
        # to ``eta``. The first zero is the physical ``msini``.
        msini = abs(np.roots([1, -eta, -2 * eta, -eta])[-1])

        # Compute the semi-major axis in km and convert to AU
        semi_a = np.sqrt(grav * msini * 10 ** self.guess['log_period'] /
                         10 ** self.guess['log_k'] / (2 * np.pi) /
                         np.sqrt(1. - ecc) ** 2)
        return msini, semi_a

    # The probability
    def lnprob(self, theta_list):
        """
        This function calculates the ln of the probabilities to be used in the
        MCMC estimation.

        Parameters
        ----------
        theta: sequence

        Returns
        -------
        The probability of the signal rv being the result of a model with the
        parameters theta
        """
        # The common parameters
        theta = {'log_k': theta_list[0],
                 'log_period': theta_list[1],
                 't0': theta_list[2]}

        # Parametrization option-specific parameters
        if self.parametrization == 'mc10':
            theta['omega'] = theta_list[3]
            theta['log_ecc'] = theta_list[4]
        elif self.parametrization == 'exofast':
            theta['sqe_cosw'] = theta_list[3]
            theta['sqe_sinw'] = theta_list[4]

        # Instrumental parameters
        for i in range(self.n_ds):
            theta['gamma_{}'.format(i)] = theta_list[5 + i]
            if self.use_add_sigma is True:
                theta['sigma_{}'.format(i)] = theta_list[5 + self.n_ds + i]

        lp = prior.flat(theta, self.bounds)
        params = self.prepare_params(theta, self.bounds)
        if not np.isfinite(lp):
            return -np.inf
        return lp - 0.5 * self.lnlike(params)

    # Using emcee to estimate the orbital parameters
    def emcee_orbit(self, nwalkers=20, nsteps=1000, p_scale=2.0, nthreads=1,
                    ballsizes=None):
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

        ballsizes : ``dict``
            The one-dimensional size of the volume from which to generate a
            first position to start the chain.

        Returns
        -------
        sampler : ``emcee.EnsembleSampler``
            The resulting sampler object.
        """
        ndim = len(self.keys)

        ballsizes = 1E-4
        pos = [self.guess['log_k'].value, self.guess['log_period'].value,
               self.guess['t0'].value]
        if self.parametrization == 'mc10':
            pos.append(self.guess['omega'].value)
            pos.append(self.guess['log_ecc'])
        elif self.parametrization == 'exofast':
            pos.append(self.guess['sqe_cosw'])
            pos.append(self.guess['sqe_sinw'])
        for i in range(self.n_ds):
            pos.append(self.guess['gamma_{}'.format(i)].value)
            if self.use_add_sigma is True:
                pos.append(self.guess['sigma_{}'.format(i)].value)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                        a=p_scale)
        sampler.run_mcmc(pos, nsteps)
        self.sampler = sampler


if __name__ == '__main__':

    harps = dataset.RVDataSet('tests/HIP19911_HARPS.dat', t_offset=-2.45E6,
                              rv_offset='subtract_mean',
                              instrument_name='HARPS',
                              target_name='HIP 19911')
    hires = dataset.RVDataSet('tests/HIP19911_HIRES.dat', t_offset=-2.45E6,
                              rv_offset='subtract_mean',
                              instrument_name='HIRES',
                              rv_unit=u.m / u.s, target_name='HIP 19911')
    #_datasets = [harps, hires]
    _datasets = [hires]

    _guess = {'log_k': np.log10(400),
              'log_period': 3.0791812,
              #'log_k': 0.89093152,
              #'log_period': 3.3165404,
              't0': 4629.04013,
              'omega': 41.2891974 * np.pi / 180,
              'log_ecc': -3,
              #'sqe_cosw': np.sqrt(.822429) * np.cos(41.2891974 * u.deg),
              #'sqe_sinw': np.sqrt(.822429) * np.sin(41.2891974 * u.deg),
              #'gamma_0': 0,
              'gamma_0': -3.86723446}#,
              #'sigma_0': 0.001 * u.km / u.s,
              #'sigma_1': 0.001 * u.km / u.s}

    _bounds = {'log_k': (2, 3),
               'log_period': (2.8, 3.2),
               't0': (4000, 5000),
               'omega': (0, 100 * np.pi / 180),
               'log_ecc': (-0.09, -0.08),
               #'gamma_0': (-1, 1) * u.km / u.s}#,
               'gamma_0': (-4000, -3000)}#,
               #'sigma_0': (0.0001, 0.5) * u.km / u.s,
               #'sigma_1': (0.0001, 0.5) * u.km / u.s}

    estim = FullOrbit(_datasets, _guess, _bounds, use_add_sigma=False,
                      parametrization='mc10')
    estim.lmfit_orbit()
    _msini, _semi_a = estim.compute_dynamics(main_body_mass=0.987)
    print(_msini, _semi_a)
    #_theta = [0.89093152, 3.3165404, 4629.04013, 41.2891974, -0.08490158,
    #          -3.86723446]
    #estim.emcee_orbit(nthreads=1)
    #corner.corner(res.flatchain, truths=list(res.params.valuesdict().values()))
