#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

"""
This module contains the different priors to be used with ``keppy``.
"""


# Flat priors
def flat(theta, parametrization):
    """
    Computes a flat prior probability for a given set of parameters theta.

    Parameters
    ----------
    theta : ``dict``
        The orbital and instrumental parameters. This dictionary must have
        keywords identical to the parameter names (which depend on the number
        of data sets, the parametrization option and if the additional
        uncertainties are also being estimated.

    Returns
    -------
    prob : ``float``
        The prior probability for a given set of orbital and instrumental
        parameters.
    """
    # Compute the eccentricity beforehand to impose a prior of e < 1 on it
    if parametrization == 'mc10':
        ecc = theta['ecc']
        omega = theta['omega']
    elif parametrization == 'exofast':
        ecc = theta['sqe_cosw'] ** 2 + theta['sqe_sinw'] ** 2
        omega = np.arctan2(theta['sqe_sinw'], theta['sqe_cosw'])
    else:
        raise ValueError('The parametrization has to be either "mc10" or '
                         '"exofast".')

    check = []

    # Eccentricity must be between 0 and 1
    if 0 < ecc < 1:
        check.append(True)
    else:
        check.append(False)

    # sqe_cosw and sqe_sinw must be between -1 and 1
    if parametrization == 'exofast':
        if -1 < theta['sqe_cosw'] < 1 and -1 < theta['sqe_sinw'] < 1:
            check.append(True)
        else:
            check.append(False)

    # omega must be between -pi and pi
    if -np.pi < omega < np.pi:
        check.append(True)
    else:
        check.append(False)

    if all(check) is True:
        prob = 0.0
    else:
        prob = -np.inf

    return prob
