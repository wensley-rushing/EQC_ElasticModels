# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:36:58 2023

@author: Uzo Uwaoma - udu@uw.edu
"""
import numpy as np


def nz_horiz_seismic_shear(spectral_shape_factor, hazard_factor, return_per_factor_sls,
                           return_per_factor_uls, fault_factor, perform_factor, ductility_factor,
                           seismic_weight):

    """
    This function computes the horizontal sesimic shear force per NZS 1170-5:2004.


    Parameters
    ----------
    spectral_shape_factor : float
        Spectral shape factor Ch(T), obtained from Table 3.1.
    hazard_factor : float
        Hazard factor Z, obtained from Clause 3.1.4.
    return_per_factor : float
        Return period factor R, obtained from Clause 3.1.5.
    fault_factor : float
        Near-fault factor N(T,D), obtained from Clause 3.1.6.
    perform_factor : float
        Structural Performance factor Sp, obtained from Clause 4.4.
    ductility_factor : float
        Ductility factor Ku, obtained from
    seismic_weight: float
        Seismic weight of structure

    Returns
    -------
    horiz_sesimic_shear : float
        The horizontal sesimic shear, V, acting at tha base of the structure.

    """
    elastic_spectrum_ordinate = (spectral_shape_factor * hazard_factor *
                                 return_per_factor_uls * fault_factor)  # C(T1); Clause 3.1.1

    horiz_design_action_coeff = (elastic_spectrum_ordinate * perform_factor /
                                 ductility_factor)  # Cd(T1); Clause 5.2.1.1

    action_coeff_lower_lim1 = (hazard_factor/20 + 0.02)*return_per_factor_uls  # Eqn. 5.2(2)
    action_coeff_lower_lim2 = 0.03 * return_per_factor_uls

    horiz_design_action_coeff = max(horiz_design_action_coeff, action_coeff_lower_lim1, action_coeff_lower_lim2)

    horiz_sesimic_shear = horiz_design_action_coeff * seismic_weight  # V; Clause 6.2.1

    return round(horiz_sesimic_shear, 2)



def nz_horiz_force_distribution(total_base_shear, story_weights, story_heights):
    """
    Computes the horizonatal force at each story of a building for use in ELF analysis.


    Parameters
    ----------
    total_base_shear : float
        Total base shear of building
    story_weights : 1D numpy array
        Array of story weights arranged from the first floor to the topmost floor.
    story_heights : 1D numpy array
        Array of cummulative story heights from the first floor to the topmost floor.

    Returns
    -------
    equiv_static_force : 1D numpy array
        Equivalent static force on each story

    """

    equiv_static_force = np.zeros(len(story_weights))
    weights_dot_heights = story_weights.dot(story_heights)

    for ii in range(len(equiv_static_force)):
        if ii == len(equiv_static_force) - 1:
            ft = 0.08 * total_base_shear
        else:
            ft = 0

        equiv_static_force[ii] = (ft +
                                  (0.92 * total_base_shear * story_weights[ii] * story_heights[ii] / weights_dot_heights))

    return equiv_static_force

