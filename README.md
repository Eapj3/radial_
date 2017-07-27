# radial
A pure Python code for estimating radial velocities of stars with a massive companion.

[Read the docs here!](http://radial.rtfd.io/)

### Installation

In order to install `radial`, use the following command:

```
python setup.py install
```

**Note:** In order to compile the documentation into `html` files, you will need `sphinx` and `sphinx_rtd_theme`. To compile it, navigate to the `docs` folder and issue the following command:

```
make html
```

### Changelog

##### Version 0.4: 
* ``lmfit`` is used to perform minimization, and ``emcee`` to estimate uncertainties. The support for ``astropy.units`` on parameter estimation was dropped because of performance issues.
* The project is now called ``radial``.

##### Version 0.5:
* The project now has proper documentation.

### License

Copyright 2017 Leonardo A. dos Santos. `radial` is a free software available under the MIT License. For details see the LICENSE file.