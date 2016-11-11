#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.table as t
import astropy.units as u

"""
This module is used to manage radial velocities datasets.
"""


# The RV dataset class
class RVDataSet(object):
    """

    Parameters
    ----------
    file
    t_col
    rv_col
    rv_unc_col
    skiprows
    delimiter
    t_offset
    rv_offset
    t_unit
    rv_unit
    instrument_name
    """
    def __init__(self, file, t_col=0, rv_col=1, rv_unc_col=None, skiprows=0,
                 delimiter=None, t_offset=None, rv_offset=None, t_unit=None,
                 rv_unit=None, instrument_name=None):

        # Setting up the default values
        # The time unit
        if t_unit is None:
            self.t_unit = u.d
        else:
            self.t_unit = t_unit
        # The RV unit
        if rv_unit is None:
            self.rv_unit = u.km / u.s
        else:
            self.rv_unit = rv_unit

        # The time offset
        if t_offset is None:
            self.t_offset = 0
        elif isinstance(t_offset, u.Quantity) is True:
            self.t_offset = t_offset
        else:
            self.t_offset = t_offset * self.t_unit
        # The RV offset
        if rv_offset is None:
            self.rv_offset = 0
        elif isinstance(rv_offset, u.Quantity) is True:
            self.rv_offset = rv_offset
        else:
            self.rv_offset = rv_offset * self.rv_unit

        # Read the data from file
        self.t = np.loadtxt(file, usecols=(t_col,), skiprows=skiprows,
                            delimiter=delimiter) * self.t_unit + self.t_offset
        self.rv = np.loadtxt(file, usecols=(rv_col,), skiprows=skiprows,
                             delimiter=delimiter) * self.rv_unit + \
            self.rv_offset
        if rv_unc_col is None:
            self.rv_unc = None
        else:
            self.rv_unc = np.loadtxt(file, usecols=(rv_unc_col,),
                                     skiprows=skiprows, delimiter=delimiter) \
                * self.rv_unit

        # Create an astropy table with the data
        self.table = t.Table([self.t, self.rv, self.rv_unc],
                             names=['Time', 'RV', 'RV sigma'],
                             meta={'Instrument': instrument_name})
