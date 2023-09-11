# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:03:42 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import numpy as np


# The data below is based on NZS 1170.5:2004 - Table 3.1
periods = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
               0.6, 0.7, 0.8, 0.9, 1.0, 1.5,
               2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

spect_shape_fac = np.array([2.36, 2.36, 2.36, 2.36, 2.36, 2.00,
                 1.74, 1.55, 1.41, 1.29, 1.19, 0.88,
                 0.66, 0.53, 0.44, 0.32, 0.25, 0.20])


def spectral_shape_fac(nat_period):
    ch_t = np.interp(nat_period, periods, spect_shape_fac)

    return ch_t



