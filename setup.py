#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import setup

with open("README.md","r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = "extended-kalman-filter",
    version = "1.0.0",
    author = "Marcin Kuropatwiński",
    author_email = "marcin@talking2rabbit.com",
    description = ("Fast implementation of the extended Kalman filter with STP modelling of speech and noise processes."),
    license = "GNU 3.0",
    packages = ["kalman"],
    long_description_content_type="text/markdown",
    long_description=long_description,
)
