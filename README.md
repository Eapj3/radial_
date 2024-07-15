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

#### scripts

Files in the `scripts` folder are used separately to help with data extraction; to use them from any location you may save them into `/usr/local/bin` or `$HOME/.local/bin` directories.

### Changelog

##### Version 0.4: 
* ``lmfit`` is used to perform minimization, and ``emcee`` to estimate uncertainties. The support for ``astropy.units`` on parameter estimation was dropped because of performance issues.
* The project is now called ``radial``.

##### Version 0.5:
* The project now has proper documentation.

##### Version FDU:

- Fixed compatibility issues with newer versions of python and newer versions of the libraries and packages required.
- Added new scripts for data extraction and pre-processing.
- Fixed `emcee` parallelization using `ThreadpoolController` and`Pool`.
  - `nthreads` is no longer a valid argument for `FullOrbit.emcee_orbit` method. `threads_per_worker` must be used instead; for more details about usage, read the docstrirng.

### License

This project is a fork of [radial](https://github.com/ladsantos/radial) by Leonardo dos Santos.

Copyright 2017 Leonardo A. dos Santos. `radial` is a free software available under the MIT License. For details see the LICENSE file.

- Original work is licensed under the MIT license. See the LICENSE file for details.
- Modifications and new code by Edgar Pérez are licensed under the MIT license. See LICENSE file for details.