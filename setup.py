#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

setup(
    name="radial",
    version="0.5a",
    author="Leonardo dos Santos",
    author_email="leonardoags@usp.br",
    packages=["radial"],
    url="https://github.com/RogueAstro/radial",
    license="MIT",
    description="Radial velocities analysis with Python",
    install_requires=["numpy", "scipy", "matplotlib", "emcee", "corner",
                      "lmfit", "astropy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ]
)
