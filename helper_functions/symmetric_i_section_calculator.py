# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:50:34 2023

@author: Uzo Uwaoma - udu@uw.edu

Formulas adapted from https://calcresource.com/cross-section-doubletee.html
and have been verified.
"""
import numpy as np
import pandas as pd


def get_I_section_prop(flange_thick, flange_width, web_thick, depth):
    """
    Computes the area, moment of inertias, section moduli, and radius of gyration for a given I-section.

    Note that this function only works for a symmetric I-section.

    The function is unit-agnostic, so the user should the units of the arguments
    are consistent.

    Parameters
    ----------
    flange_thick : float
        flange thickness, tf.
    flange_width : float
        flange width, b.
    web_thick : float
        web thickness, tw.
    depth : float
        overall height of section, d.

    Returns
    -------
    section_prop : pandas Series
        A series containing the area, moment of inertias, section moduli, and radius of gyration.

    """

    web_depth = depth - 2*flange_thick

    area = (2 * flange_width * flange_thick) + (web_depth * web_thick)

    mom_inertia_x = (flange_width * depth**3 / 12) - ((flange_width - web_thick) * web_depth**3 / 12)
    mom_inertia_y = (flange_thick * flange_width**3 / 6) + (web_depth * web_thick**3 / 12)
    mom_inertia_z = mom_inertia_x + mom_inertia_y

    y_centroid = 0.5 * depth
    x_centroid = 0.5 * flange_width

    elastic_section_mod_x = mom_inertia_x / y_centroid
    elastic_section_mod_y = mom_inertia_y / x_centroid

    plastic_section_mod_x = (flange_width * depth**2 / 4) - ((flange_width - web_thick) * web_depth**2 / 4)
    plastic_section_mod_y = (flange_thick * flange_width**2 / 2) + (web_depth * web_thick**2 / 4)

    rad_gyr_x = np.sqrt(mom_inertia_x / area)
    rad_gyr_y = np.sqrt(mom_inertia_y / area)

    section_prop = pd.Series([area, mom_inertia_x, mom_inertia_y, mom_inertia_z,
                              elastic_section_mod_x, elastic_section_mod_y,
                              plastic_section_mod_x, plastic_section_mod_y,
                              rad_gyr_x, rad_gyr_y],
                     index=['A', 'Ix', 'Iy', 'Iz', 'Sx', 'Sy', 'Zx', 'Zy', 'rx', 'ry']).round(2)

    return section_prop


if __name__ == "__main__":
    sample_prop = get_I_section_prop(2.17, 16.1, 1.22, 44.8)
