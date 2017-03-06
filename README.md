# radial
A pure Python code for estimating radial velocities of stars with a massive companion.

The new 0.4 version uses ``lmfit`` to perform minimization, and ``emcee`` to estimate uncertainties. The support for ``astropy.units`` on parameter estimation was dropped because of performance issues.

As of version 0.4, the project is now called ``radial``.