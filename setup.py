#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read(

setup(
    name = "Kalman filter with STP parameters for speech and noise"),
    version = "1.0.0",
    author = "Marcin Kuropatwiński",
    author_email = "marcin@talking2rabbit.com",
    description = ("Fast implementation of the extended Kalman filter with STP modelling of speech and noise processes."),
    license = "GNU 3.0",
    packages = ["kalman"],
    long_desription = read('README.md'),
    )

