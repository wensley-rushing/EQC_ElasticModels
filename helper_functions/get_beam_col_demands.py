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

    force_x = peak_total_resp[0::6]
    force_y = peak_total_resp[1::6]
    force_z = peak_total_resp[2::6]

    mom_x = peak_total_resp[3::6]
    mom_y = peak_total_resp[4::6]
    mom_z = peak_total_resp[5::6]

    max_force_x = force_x.max()
    max_force_y = force_y.max()
    max_force_z = force_z.max()

    max_mom_x = mom_x.max()
    max_mom_y = mom_y.max()
    max_mom_z = mom_z.max()


    return max_force_x, max_force_y, max_force_z, max_mom_x, max_mom_y, max_mom_z


def process_beam_col_resp(elem_type, beam_col_resp_folder, angular_freq, damp_ratio, num_modes):
    """
    Computes the maximum beam shear force and bending moment for each floor.


    Parameters
    ----------
    elem_type : string
        Type of element whose data is to be processed.
            Can be 'beam', 'column', 'wall'
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
    pk_modal_resp_flr_01 = np.loadtxt(beam_col_resp_folder + 'floor01_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_02 = np.loadtxt(beam_col_resp_folder + 'floor02_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_03 = np.loadtxt(beam_col_resp_folder + 'floor03_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_04 = np.loadtxt(beam_col_resp_folder + 'floor04_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_05 = np.loadtxt(beam_col_resp_folder + 'floor05_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_06 = np.loadtxt(beam_col_resp_folder + 'floor06_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_07 = np.loadtxt(beam_col_resp_folder + 'floor07_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_08 = np.loadtxt(beam_col_resp_folder + 'floor08_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_09 = np.loadtxt(beam_col_resp_folder + 'floor09_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_10 = np.loadtxt(beam_col_resp_folder + 'floor10_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_11 = np.loadtxt(beam_col_resp_folder + 'floor11_' + elem_type + 'Resp.txt')

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
    Fx_flr_01, Fy_flr_01, Fz_flr_01, Mx_flr_01, My_flr_01, Mz_flr_01 = get_max_shear_and_moment(pk_total_resp_flr_01)
    Fx_flr_02, Fy_flr_02, Fz_flr_02, Mx_flr_02, My_flr_02, Mz_flr_02 = get_max_shear_and_moment(pk_total_resp_flr_02)
    Fx_flr_03, Fy_flr_03, Fz_flr_03, Mx_flr_03, My_flr_03, Mz_flr_03 = get_max_shear_and_moment(pk_total_resp_flr_03)
    Fx_flr_04, Fy_flr_04, Fz_flr_04, Mx_flr_04, My_flr_04, Mz_flr_04 = get_max_shear_and_moment(pk_total_resp_flr_04)
    Fx_flr_05, Fy_flr_05, Fz_flr_05, Mx_flr_05, My_flr_05, Mz_flr_05 = get_max_shear_and_moment(pk_total_resp_flr_05)
    Fx_flr_06, Fy_flr_06, Fz_flr_06, Mx_flr_06, My_flr_06, Mz_flr_06 = get_max_shear_and_moment(pk_total_resp_flr_06)
    Fx_flr_07, Fy_flr_07, Fz_flr_07, Mx_flr_07, My_flr_07, Mz_flr_07 = get_max_shear_and_moment(pk_total_resp_flr_07)
    Fx_flr_08, Fy_flr_08, Fz_flr_08, Mx_flr_08, My_flr_08, Mz_flr_08 = get_max_shear_and_moment(pk_total_resp_flr_08)
    Fx_flr_09, Fy_flr_09, Fz_flr_09, Mx_flr_09, My_flr_09, Mz_flr_09 = get_max_shear_and_moment(pk_total_resp_flr_09)
    Fx_flr_10, Fy_flr_10, Fz_flr_10, Mx_flr_10, My_flr_10, Mz_flr_10 = get_max_shear_and_moment(pk_total_resp_flr_10)
    Fx_flr_11, Fy_flr_11, Fz_flr_11, Mx_flr_11, My_flr_11, Mz_flr_11 = get_max_shear_and_moment(pk_total_resp_flr_11)

    # Initialize dataframe to save maximum values
    max_beam_demands = pd.DataFrame()

    max_beam_demands['Floor'] = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']

    max_beam_demands['Fx (kN)'] = [Fx_flr_01, Fx_flr_02, Fx_flr_03, Fx_flr_04, Fx_flr_05,
                                   Fx_flr_06, Fx_flr_07, Fx_flr_08, Fx_flr_09, Fx_flr_10,
                                   Fx_flr_11]

    max_beam_demands['Fy (kN)'] = [Fy_flr_01, Fy_flr_02, Fy_flr_03, Fy_flr_04, Fy_flr_05,
                                   Fy_flr_06, Fy_flr_07, Fy_flr_08, Fy_flr_09, Fy_flr_10,
                                   Fy_flr_11]

    max_beam_demands['Fz (kN)'] = [Fz_flr_01, Fz_flr_02, Fz_flr_03, Fz_flr_04, Fz_flr_05,
                                   Fz_flr_06, Fz_flr_07, Fz_flr_08, Fz_flr_09, Fz_flr_10,
                                   Fz_flr_11]

    max_beam_demands['Mx (kN-m)'] = [Mx_flr_01, Mx_flr_02, Mx_flr_03, Mx_flr_04, Mx_flr_05,
                                   Mx_flr_06, Mx_flr_07, Mx_flr_08, Mx_flr_09, Mx_flr_10,
                                   Mx_flr_11]

    max_beam_demands['My (kN-m)'] = [My_flr_01, My_flr_02, My_flr_03, My_flr_04, My_flr_05,
                                   My_flr_06, My_flr_07, My_flr_08, My_flr_09, My_flr_10,
                                   My_flr_11]

    max_beam_demands['Mz (kN-m)'] = [Mz_flr_01, Mz_flr_02, Mz_flr_03, Mz_flr_04, Mz_flr_05,
                                   Mz_flr_06, Mz_flr_07, Mz_flr_08, Mz_flr_09, Mz_flr_10,
                                   Mz_flr_11]

    return(max_beam_demands)


