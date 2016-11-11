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
    Read and store the data and information of the radial velocities dataset in
    an intelligible manner. This class utilizes the power and convenience of
    ``astropy`` units and tables.

    Parameters
    ----------
    file : ``str``
        Name of the file that contains the radial velocities data.

    t_col : ``int``, optional
        Column number of the data file that corresponds to time. Default is 0.

    rv_col : ``int``, optional
        Column number of the data file that corresponds to radial velocities.
        Default is 1.

    rv_unc_col : ``int``, optional
        Column number of the data file that corresponds to radial velocity
        uncertainties. Default is 2.

    skiprows : ``int``, optional
        Number of rows to skip from the data file. Default is 0.

    delimiter : ``str`` or ``None``, optional
        String that is used to separate the columns in the data file. If
        ``None``, uses the default value from ``numpy.loadtxt``. Default is
        ``None``.

    t_offset : ``float``, ``astropy.units.Quantity`` or ``None``, optional
        Numerical offset to be summed to the time array. If ``None``, no offset
        is applied. Default is ``None``.

    rv_offset : ``float``, ``astropy.units.Quantity`` or ``None``, optional
        Numerical offset to be summed to the radial velocities array. If
        ``None``, no offset is applied. Default is ``None``.

    t_unit : ``astropy.units`` or ``None``, optional
        The unit of the time array, in ``astropy.units``. If ``None``, uses
        days. Default is ``None``.

    rv_unit : ``astropy.units`` or ``None``, optional
        The unit of the radial velocities and uncertainties arrays, in
        ``astropy.units``. If ``None``, uses km/s. Default is ``None``.

    instrument_name : ``str`` or ``None``, optional
        Name of the instrument, which will be save in the metadata. Default is
        ``None``.
    """
    def __init__(self, file, t_col=0, rv_col=1, rv_unc_col=2, skiprows=0,
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
        self.rv_unc = np.loadtxt(file, usecols=(rv_unc_col,), skiprows=skiprows,
                                 delimiter=delimiter) * self.rv_unit

        # Create an astropy table with the data
        self.table = t.Table([self.t, self.rv, self.rv_unc],
                             names=['Time', 'RV', 'RV sigma'],
                             meta={'Instrument': instrument_name})
