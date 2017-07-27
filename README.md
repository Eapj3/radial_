# radial
A pure Python code for estimating radial velocities of stars with a massive companion.

[Read the docs here!](http://radial.rtfd.io/)

The new 0.4 version uses ``lmfit`` to perform minimization, and ``emcee`` to estimate uncertainties. The support for ``astropy.units`` on parameter estimation was dropped because of performance issues.

As of version 0.4, the project is now called ``radial``.

In order to install `radial`, use the following command:

```
python setup.py install
```

**Note:** In order to compile the documentation into `html` files, you will need `sphinx` and `sphinx_rtd_theme`. To compile it, navigate to the `docs` folder and issue the following command:

```
make html
```