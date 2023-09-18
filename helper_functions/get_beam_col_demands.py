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


def get_max_shear_and_moment(total_mrsa_resp, pos_torsion_resp, neg_torsion_resp):
    """
    This functions gives the maximum shear and bending moment for all beams/columns on a floor.

    It takes in the peak total response from a MRSA, together with the demands from
    the static analysis for accidental torsion for positive and negative eccentricity.

    It combines the demands from the MRSA with the two static analyses and selects the most
    onerous of the two combinations.

    Taking cognizance of how OpenSeesPy recorders for beam-column elements force response
    are stored, it then indexes into combined reposne to extract the force and moment values.

    In a 3D model each node of a beam-column element has 6DOFs, so OpenSeesPy
    records the underlisted force response for each element:
    Fx-i, Fy-i, Fz-i, Mx-i, My-i, Mz-i, Fx-j, Fy-j, Fz-j, Mx-j, My-j, Mz-j

    Parameters
    ----------
    total_mrsa_resp : Numpy array
        Peak total response for all beam elements in a floor.
    pos_torsion_resp: Numpy array
        Demands from static analysis for accidental torsion using positive directional loading.
    neg_torsion_resp: Numpy array
        Demands from static analysis for accidental torsion using negative directional loading.

    Returns
    -------
    max_shear_force : float
        Maximum beam shear force on a floor.
    max_mom : float
        Maximum beam bending moment on a floor.

    """

    '''
    Two combinations are considered
        # 1. MRSA + Positive accidental torsional response
        # 2. MRSA + Negative accidental torsional response
    '''

    # COMBO 1: MRSA + Positive accidental torsional response
    combo_1_resp = total_mrsa_resp + pos_torsion_resp

    # COMBO 2: MRSA + Negative accidental torsional response
    combo_2_resp = total_mrsa_resp + neg_torsion_resp

    # Compare demands in both combinations and return element-wise maximums
    combined_maximum_resp = np.maximum(combo_1_resp, combo_2_resp)

    # Extract shear forces and moments
    force_x = combined_maximum_resp[0::6]
    force_y = combined_maximum_resp[1::6]
    force_z = combined_maximum_resp[2::6]

    mom_x = combined_maximum_resp[3::6]
    mom_y = combined_maximum_resp[4::6]
    mom_z = combined_maximum_resp[5::6]

    max_force_x = force_x.max()
    max_force_y = force_y.max()
    max_force_z = force_z.max()

    max_mom_x = mom_x.max()
    max_mom_y = mom_y.max()
    max_mom_z = mom_z.max()


    return max_force_x, max_force_y, max_force_z, max_mom_x, max_mom_y, max_mom_z


def process_beam_col_resp(elem_type, mrsa_resp_folder, pos_torsion_resp_folder, neg_torsion_resp_folder, angular_freq, damp_ratio, num_modes, elf_mrsa_scale_factor):
    """
    Computes the maximum beam shear force and bending moment for each floor.


    Parameters
    ----------
    elem_type : string
        Type of element whose data is to be processed.
            Can be 'beam', 'column', 'wall'
    mrsa_resp_folder : string
        Directory containing the peak modal response from MRSA for all floors.
    pos_torsion_resp_folder : string
        Directory containing the demands from static analysis for accidental torsion using positive directional loading.
    neg_torsion_resp_folder : string
        Directory containing the demands from static analysis for accidental torsion using negative directional loading.
    angular_freq : list
        Angular frequencies from eigen analysis of model.
    damp_ratio : float
        Damping ratio of model.
    num_modes : int
        Number of modes to use in computing peak total response.
    elf_mrsa_scale_factor : float
        Scale factor for scaling MRSA demands to ELF.

    Returns
    -------
    max_beam_demands: DataFrame.
        Maximum beam shear force and bending moment for each floor.

    """
    # Load in peak modal response from MRSA
    pk_modal_resp_flr_01 = np.loadtxt(mrsa_resp_folder + 'floor01_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_02 = np.loadtxt(mrsa_resp_folder + 'floor02_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_03 = np.loadtxt(mrsa_resp_folder + 'floor03_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_04 = np.loadtxt(mrsa_resp_folder + 'floor04_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_05 = np.loadtxt(mrsa_resp_folder + 'floor05_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_06 = np.loadtxt(mrsa_resp_folder + 'floor06_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_07 = np.loadtxt(mrsa_resp_folder + 'floor07_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_08 = np.loadtxt(mrsa_resp_folder + 'floor08_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_09 = np.loadtxt(mrsa_resp_folder + 'floor09_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_10 = np.loadtxt(mrsa_resp_folder + 'floor10_' + elem_type + 'Resp.txt')
    pk_modal_resp_flr_11 = np.loadtxt(mrsa_resp_folder + 'floor11_' + elem_type + 'Resp.txt')

    # Compute peak total response from MRSA by CQC combination and ampify Velf / Vmrsa
    pk_total_resp_flr_01 = modal_combo(pk_modal_resp_flr_01, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor
    pk_total_resp_flr_02 = modal_combo(pk_modal_resp_flr_02, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor
    pk_total_resp_flr_03 = modal_combo(pk_modal_resp_flr_03, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor
    pk_total_resp_flr_04 = modal_combo(pk_modal_resp_flr_04, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor
    pk_total_resp_flr_05 = modal_combo(pk_modal_resp_flr_05, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor
    pk_total_resp_flr_06 = modal_combo(pk_modal_resp_flr_06, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor
    pk_total_resp_flr_07 = modal_combo(pk_modal_resp_flr_07, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor
    pk_total_resp_flr_08 = modal_combo(pk_modal_resp_flr_08, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor
    pk_total_resp_flr_09 = modal_combo(pk_modal_resp_flr_09, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor
    pk_total_resp_flr_10 = modal_combo(pk_modal_resp_flr_10, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor
    pk_total_resp_flr_11 = modal_combo(pk_modal_resp_flr_11, angular_freq, damp_ratio, num_modes) * elf_mrsa_scale_factor

    # Load in results from static accidental torsion analysis
    # Positive loading
    pos_torsion_resp_flr_01 = np.loadtxt(pos_torsion_resp_folder + 'floor01_' + elem_type + 'Resp.txt')
    pos_torsion_resp_flr_02 = np.loadtxt(pos_torsion_resp_folder + 'floor02_' + elem_type + 'Resp.txt')
    pos_torsion_resp_flr_03 = np.loadtxt(pos_torsion_resp_folder + 'floor03_' + elem_type + 'Resp.txt')
    pos_torsion_resp_flr_04 = np.loadtxt(pos_torsion_resp_folder + 'floor04_' + elem_type + 'Resp.txt')
    pos_torsion_resp_flr_05 = np.loadtxt(pos_torsion_resp_folder + 'floor05_' + elem_type + 'Resp.txt')
    pos_torsion_resp_flr_06 = np.loadtxt(pos_torsion_resp_folder + 'floor06_' + elem_type + 'Resp.txt')
    pos_torsion_resp_flr_07 = np.loadtxt(pos_torsion_resp_folder + 'floor07_' + elem_type + 'Resp.txt')
    pos_torsion_resp_flr_08 = np.loadtxt(pos_torsion_resp_folder + 'floor08_' + elem_type + 'Resp.txt')
    pos_torsion_resp_flr_09 = np.loadtxt(pos_torsion_resp_folder + 'floor09_' + elem_type + 'Resp.txt')
    pos_torsion_resp_flr_10 = np.loadtxt(pos_torsion_resp_folder + 'floor10_' + elem_type + 'Resp.txt')
    pos_torsion_resp_flr_11 = np.loadtxt(pos_torsion_resp_folder + 'floor11_' + elem_type + 'Resp.txt')

    # Negative loading
    neg_torsion_resp_flr_01 = np.loadtxt(neg_torsion_resp_folder + 'floor01_' + elem_type + 'Resp.txt')
    neg_torsion_resp_flr_02 = np.loadtxt(neg_torsion_resp_folder + 'floor02_' + elem_type + 'Resp.txt')
    neg_torsion_resp_flr_03 = np.loadtxt(neg_torsion_resp_folder + 'floor03_' + elem_type + 'Resp.txt')
    neg_torsion_resp_flr_04 = np.loadtxt(neg_torsion_resp_folder + 'floor04_' + elem_type + 'Resp.txt')
    neg_torsion_resp_flr_05 = np.loadtxt(neg_torsion_resp_folder + 'floor05_' + elem_type + 'Resp.txt')
    neg_torsion_resp_flr_06 = np.loadtxt(neg_torsion_resp_folder + 'floor06_' + elem_type + 'Resp.txt')
    neg_torsion_resp_flr_07 = np.loadtxt(neg_torsion_resp_folder + 'floor07_' + elem_type + 'Resp.txt')
    neg_torsion_resp_flr_08 = np.loadtxt(neg_torsion_resp_folder + 'floor08_' + elem_type + 'Resp.txt')
    neg_torsion_resp_flr_09 = np.loadtxt(neg_torsion_resp_folder + 'floor09_' + elem_type + 'Resp.txt')
    neg_torsion_resp_flr_10 = np.loadtxt(neg_torsion_resp_folder + 'floor10_' + elem_type + 'Resp.txt')
    neg_torsion_resp_flr_11 = np.loadtxt(neg_torsion_resp_folder + 'floor11_' + elem_type + 'Resp.txt')

    # Extract maximum shear force and bending moment for each floor
    Fx_flr_01, Fy_flr_01, Fz_flr_01, Mx_flr_01, My_flr_01, Mz_flr_01 = get_max_shear_and_moment(pk_total_resp_flr_01, pos_torsion_resp_flr_01, neg_torsion_resp_flr_01)
    Fx_flr_02, Fy_flr_02, Fz_flr_02, Mx_flr_02, My_flr_02, Mz_flr_02 = get_max_shear_and_moment(pk_total_resp_flr_02, pos_torsion_resp_flr_02, neg_torsion_resp_flr_02)
    Fx_flr_03, Fy_flr_03, Fz_flr_03, Mx_flr_03, My_flr_03, Mz_flr_03 = get_max_shear_and_moment(pk_total_resp_flr_03, pos_torsion_resp_flr_03, neg_torsion_resp_flr_03)
    Fx_flr_04, Fy_flr_04, Fz_flr_04, Mx_flr_04, My_flr_04, Mz_flr_04 = get_max_shear_and_moment(pk_total_resp_flr_04, pos_torsion_resp_flr_04, neg_torsion_resp_flr_04)
    Fx_flr_05, Fy_flr_05, Fz_flr_05, Mx_flr_05, My_flr_05, Mz_flr_05 = get_max_shear_and_moment(pk_total_resp_flr_05, pos_torsion_resp_flr_05, neg_torsion_resp_flr_05)
    Fx_flr_06, Fy_flr_06, Fz_flr_06, Mx_flr_06, My_flr_06, Mz_flr_06 = get_max_shear_and_moment(pk_total_resp_flr_06, pos_torsion_resp_flr_06, neg_torsion_resp_flr_06)
    Fx_flr_07, Fy_flr_07, Fz_flr_07, Mx_flr_07, My_flr_07, Mz_flr_07 = get_max_shear_and_moment(pk_total_resp_flr_07, pos_torsion_resp_flr_07, neg_torsion_resp_flr_07)
    Fx_flr_08, Fy_flr_08, Fz_flr_08, Mx_flr_08, My_flr_08, Mz_flr_08 = get_max_shear_and_moment(pk_total_resp_flr_08, pos_torsion_resp_flr_08, neg_torsion_resp_flr_08)
    Fx_flr_09, Fy_flr_09, Fz_flr_09, Mx_flr_09, My_flr_09, Mz_flr_09 = get_max_shear_and_moment(pk_total_resp_flr_09, pos_torsion_resp_flr_09, neg_torsion_resp_flr_09)
    Fx_flr_10, Fy_flr_10, Fz_flr_10, Mx_flr_10, My_flr_10, Mz_flr_10 = get_max_shear_and_moment(pk_total_resp_flr_10, pos_torsion_resp_flr_10, neg_torsion_resp_flr_10)
    Fx_flr_11, Fy_flr_11, Fz_flr_11, Mx_flr_11, My_flr_11, Mz_flr_11 = get_max_shear_and_moment(pk_total_resp_flr_11, pos_torsion_resp_flr_11, neg_torsion_resp_flr_11)

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


