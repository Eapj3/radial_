#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

"""
This module contains the different priors to be used with ``keppy``.
"""


# Flat priors
def flat(theta, bounds):
    """
    Computes a flat prior probability for a given set of parameters theta.

    Parameters
    ----------
    theta : ``dict``
        The orbital and instrumental parameters. This dictionary must have
        keywords identical to the parameter names (which depend on the number
        of data sets, the parametrization option and if the additional
        uncertainties are also being estimated.

    bounds : ``dict``
        Bounds for the priors. This dictionary has the same keywords as
        ``theta``. Each entry has to be a sequence of size two, in which the
        first is the lower bound and the second is the higher bound. Example:
        ``bounds = {'log_k': (-1, 1), 'log_period': (1, 4)}``.

    Returns
    -------
    prob : ``float``
        The prior probability for a given set of orbital and instrumental
        parameters.
    """
    keys = theta.keys()

    # Compute the eccentricity beforehand to impose a prior of e < 1 on it
    try:
        ecc = 10 ** theta['log_ecc']
    except KeyError:
        try:
            ecc = (theta['sqe_cosw'] ** 2 + theta['sqe_sinw'] ** 2).value
        except AttributeError:
            ecc = theta['sqe_cosw'] ** 2 + theta['sqe_sinw'] ** 2

    check = [bounds[key][0] < theta[key] < bounds[key][1] for key in keys]
    if all(check) is True and ecc < 1:
        prob = 0.0
    else:
        prob = -np.inf
    return prob
