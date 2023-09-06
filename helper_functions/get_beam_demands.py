# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:07:11 2023

@author: Uzo Uwaoma - udu@uw.edu
"""
import sys
import numpy as np
import pandas as pd

# Append directory of helper functions to Pyhton Path
sys.path.append('./')

from helper_functions.cqc_modal_combo import modal_combo


def get_max_shear_and_moment(peak_total_resp):
    """
    This functions gives the maximum shear and bending moment for all beams on a floor.

    It takes in the peak total response from a MRSA. Then indexes into the shear force
    and moment values, taking cognizance of how OpenSeesPy recorders for beam-column
    elements force response are stored.

    In a 3D model each node of a beam-column element has 6DOFs, so OpenSeesPy
    records the underlisted force response for each element:
    Fx-i, Fy-i, Fz-i, Mx-i, My-i, Mz-i, Fx-j, Fy-j, Fz-j, Mx-j, My-j, Mz-j

    Parameters
    ----------
    peak_total_resp : Numpy array
        Peak total response for all beam elements in a floor.

    Returns
    -------
    max_shear_force : float
        Maximum beam shear force on a floor.
    max_mom : float
        Maximum beam bending moment on a floor.

    """

    # Extract shear forces and moments
    shear_force = peak_total_resp[2::6]

    mom_x = peak_total_resp[3::6]
    mom_y = peak_total_resp[4::6]
    mom_z = peak_total_resp[5::6]

    max_shear_force = shear_force.max()
    max_mom = max(mom_x.max(), mom_y.max(), mom_z.max())

    return max_shear_force, max_mom


def process_beam_resp(beam_resp_folder, angular_freq, damp_ratio, num_modes):
    """
    Computes the maximum beam shear force and bending moment for each floor.


    Parameters
    ----------
    beam_resp_folder : string
        Directory containing the peak modal response for all floors
    angular_freq : list
        Angular frequencies from eigen analysis of model.
    damp_ratio : float
        Damping ratio of model.
    num_modes : int
        Number of modes to use in computing peak total response.

    Returns
    -------
    max_beam_demands: DataFrame.
        Maximum beam shear force and bending moment for each floor.

    """
    # Load in peak modal response
    pk_modal_resp_flr_01 = np.loadtxt(beam_resp_folder + 'floor01_beamResp.txt')
    pk_modal_resp_flr_02 = np.loadtxt(beam_resp_folder + 'floor02_beamResp.txt')
    pk_modal_resp_flr_03 = np.loadtxt(beam_resp_folder + 'floor03_beamResp.txt')
    pk_modal_resp_flr_04 = np.loadtxt(beam_resp_folder + 'floor04_beamResp.txt')
    pk_modal_resp_flr_05 = np.loadtxt(beam_resp_folder + 'floor05_beamResp.txt')
    pk_modal_resp_flr_06 = np.loadtxt(beam_resp_folder + 'floor06_beamResp.txt')
    pk_modal_resp_flr_07 = np.loadtxt(beam_resp_folder + 'floor07_beamResp.txt')
    pk_modal_resp_flr_08 = np.loadtxt(beam_resp_folder + 'floor08_beamResp.txt')
    pk_modal_resp_flr_09 = np.loadtxt(beam_resp_folder + 'floor09_beamResp.txt')
    pk_modal_resp_flr_10 = np.loadtxt(beam_resp_folder + 'floor10_beamResp.txt')
    pk_modal_resp_flr_11 = np.loadtxt(beam_resp_folder + 'floor11_beamResp.txt')

    # Compute peak total response
    pk_total_resp_flr_01 = modal_combo(pk_modal_resp_flr_01, angular_freq, damp_ratio, num_modes)
    pk_total_resp_flr_02 = modal_combo(pk_modal_resp_flr_02, angular_freq, damp_ratio, num_modes)
    pk_total_resp_flr_03 = modal_combo(pk_modal_resp_flr_03, angular_freq, damp_ratio, num_modes)
    pk_total_resp_flr_04 = modal_combo(pk_modal_resp_flr_04, angular_freq, damp_ratio, num_modes)
    pk_total_resp_flr_05 = modal_combo(pk_modal_resp_flr_05, angular_freq, damp_ratio, num_modes)
    pk_total_resp_flr_06 = modal_combo(pk_modal_resp_flr_06, angular_freq, damp_ratio, num_modes)
    pk_total_resp_flr_07 = modal_combo(pk_modal_resp_flr_07, angular_freq, damp_ratio, num_modes)
    pk_total_resp_flr_08 = modal_combo(pk_modal_resp_flr_08, angular_freq, damp_ratio, num_modes)
    pk_total_resp_flr_09 = modal_combo(pk_modal_resp_flr_09, angular_freq, damp_ratio, num_modes)
    pk_total_resp_flr_10 = modal_combo(pk_modal_resp_flr_10, angular_freq, damp_ratio, num_modes)
    pk_total_resp_flr_11 = modal_combo(pk_modal_resp_flr_11, angular_freq, damp_ratio, num_modes)

    # Extract maximum shear force and bending moment for each floor
    bm_V_flr_01, bm_mom_flr_01 = get_max_shear_and_moment(pk_total_resp_flr_01)
    bm_V_flr_02, bm_mom_flr_02 = get_max_shear_and_moment(pk_total_resp_flr_02)
    bm_V_flr_03, bm_mom_flr_03 = get_max_shear_and_moment(pk_total_resp_flr_03)
    bm_V_flr_04, bm_mom_flr_04 = get_max_shear_and_moment(pk_total_resp_flr_04)
    bm_V_flr_05, bm_mom_flr_05 = get_max_shear_and_moment(pk_total_resp_flr_05)
    bm_V_flr_06, bm_mom_flr_06 = get_max_shear_and_moment(pk_total_resp_flr_06)
    bm_V_flr_07, bm_mom_flr_07 = get_max_shear_and_moment(pk_total_resp_flr_07)
    bm_V_flr_08, bm_mom_flr_08 = get_max_shear_and_moment(pk_total_resp_flr_08)
    bm_V_flr_09, bm_mom_flr_09 = get_max_shear_and_moment(pk_total_resp_flr_09)
    bm_V_flr_10, bm_mom_flr_10 = get_max_shear_and_moment(pk_total_resp_flr_10)
    bm_V_flr_11, bm_mom_flr_11 = get_max_shear_and_moment(pk_total_resp_flr_11)

    # Initialize dataframe to save maximum values
    max_beam_demands = pd.DataFrame()

    max_beam_demands['Floor'] = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']

    max_beam_demands['V (kN)'] = [bm_V_flr_01, bm_V_flr_02, bm_V_flr_03, bm_V_flr_04, bm_V_flr_05,
                               bm_V_flr_06, bm_V_flr_07, bm_V_flr_08, bm_V_flr_09, bm_V_flr_10,
                               bm_V_flr_11]

    max_beam_demands['M (kN-m)'] = [bm_mom_flr_01, bm_mom_flr_02, bm_mom_flr_03, bm_mom_flr_04, bm_mom_flr_05,
                               bm_mom_flr_06, bm_mom_flr_07, bm_mom_flr_08, bm_mom_flr_09, bm_mom_flr_10,
                               bm_mom_flr_11]

    return(max_beam_demands)


