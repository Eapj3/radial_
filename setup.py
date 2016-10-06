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
    name="keppy",
    version="0.2a",
    author="Leonardo dos Santos",
    author_email="ldsantos@uchicago.edu",
    packages=["keppy"],
    url="https://github.com/RogueAstro/MAROON-X_DRS",
    license="MIT",
    description="Data reduction software for the spectrograph MAROON-X",
    install_requires=["numpy", "scipy", "emcee"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ]
)
