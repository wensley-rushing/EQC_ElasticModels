# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:45:37 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import numpy as np
import pandas as pd

def get_mrsa_wall_demands(modal_combo, results_directory, angular_freq, damping_ratio, num_modes):

    flr_1_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor01_wallResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_2_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor02_wallResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_3_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor03_wallResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_4_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor04_wallResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_5_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor05_wallResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_6_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor06_wallResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_7_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor07_wallResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_8_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor08_wallResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_9_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor09_wallResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_10_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor10_wallResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_11_wall_demands = modal_combo(np.loadtxt(results_directory + 'floor11_wallResp.txt'), angular_freq, damping_ratio, num_modes)

    force_x = pd.DataFrame(columns=['Wall_1', 'Wall_2', 'Wall_3', 'Wall_4', 'Wall_5',
                                    'Wall_6', 'Wall_7', 'Wall_8', 'Wall_9', 'Wall_10'],
                           index=['Flr_1', 'Flr_2', 'Flr_3', 'Flr_4', 'Flr_5',
                                  'Flr_6', 'Flr_7', 'Flr_8', 'Flr_9', 'Flr_10', 'Flr_11'])

    force_y = pd.DataFrame(columns=force_x.columns, index=force_x.index)
    force_z = pd.DataFrame(columns=force_x.columns, index=force_x.index)
    mom_x = pd.DataFrame(columns=force_x.columns, index=force_x.index)
    mom_y = pd.DataFrame(columns=force_x.columns, index=force_x.index)
    mom_z = pd.DataFrame(columns=force_x.columns, index=force_x.index)

    force_x.loc['Flr_1'] = flr_1_wall_demands[0::12]
    force_x.loc['Flr_2'] = flr_2_wall_demands[0::12]
    force_x.loc['Flr_3'] = flr_3_wall_demands[0::12]
    force_x.loc['Flr_4'] = flr_4_wall_demands[0::12]
    force_x.loc['Flr_5'] = flr_5_wall_demands[0::12]
    force_x.loc['Flr_6'] = flr_6_wall_demands[0::12]
    force_x.loc['Flr_7'] = flr_7_wall_demands[0::12]
    force_x.loc['Flr_8'] = flr_8_wall_demands[0::12]
    force_x.loc['Flr_9'] = flr_9_wall_demands[0::12]
    force_x.loc['Flr_10'] = flr_10_wall_demands[0::12]
    force_x.loc['Flr_11'] = flr_11_wall_demands[0::12]

    force_y.loc['Flr_1'] = flr_1_wall_demands[1::12]
    force_y.loc['Flr_2'] = flr_2_wall_demands[1::12]
    force_y.loc['Flr_3'] = flr_3_wall_demands[1::12]
    force_y.loc['Flr_4'] = flr_4_wall_demands[1::12]
    force_y.loc['Flr_5'] = flr_5_wall_demands[1::12]
    force_y.loc['Flr_6'] = flr_6_wall_demands[1::12]
    force_y.loc['Flr_7'] = flr_7_wall_demands[1::12]
    force_y.loc['Flr_8'] = flr_8_wall_demands[1::12]
    force_y.loc['Flr_9'] = flr_9_wall_demands[1::12]
    force_y.loc['Flr_10'] = flr_10_wall_demands[1::12]
    force_y.loc['Flr_11'] = flr_11_wall_demands[1::12]

    force_z.loc['Flr_1'] = flr_1_wall_demands[2::12]
    force_z.loc['Flr_2'] = flr_2_wall_demands[2::12]
    force_z.loc['Flr_3'] = flr_3_wall_demands[2::12]
    force_z.loc['Flr_4'] = flr_4_wall_demands[2::12]
    force_z.loc['Flr_5'] = flr_5_wall_demands[2::12]
    force_z.loc['Flr_6'] = flr_6_wall_demands[2::12]
    force_z.loc['Flr_7'] = flr_7_wall_demands[2::12]
    force_z.loc['Flr_8'] = flr_8_wall_demands[2::12]
    force_z.loc['Flr_9'] = flr_9_wall_demands[2::12]
    force_z.loc['Flr_10'] = flr_10_wall_demands[2::12]
    force_z.loc['Flr_11'] = flr_11_wall_demands[2::12]

    mom_x.loc['Flr_1'] = flr_1_wall_demands[3::12]
    mom_x.loc['Flr_2'] = flr_2_wall_demands[3::12]
    mom_x.loc['Flr_3'] = flr_3_wall_demands[3::12]
    mom_x.loc['Flr_4'] = flr_4_wall_demands[3::12]
    mom_x.loc['Flr_5'] = flr_5_wall_demands[3::12]
    mom_x.loc['Flr_6'] = flr_6_wall_demands[3::12]
    mom_x.loc['Flr_7'] = flr_7_wall_demands[3::12]
    mom_x.loc['Flr_8'] = flr_8_wall_demands[3::12]
    mom_x.loc['Flr_9'] = flr_9_wall_demands[3::12]
    mom_x.loc['Flr_10'] = flr_10_wall_demands[3::12]
    mom_x.loc['Flr_11'] = flr_11_wall_demands[3::12]

    mom_y.loc['Flr_1'] = flr_1_wall_demands[4::12]
    mom_y.loc['Flr_2'] = flr_2_wall_demands[4::12]
    mom_y.loc['Flr_3'] = flr_3_wall_demands[4::12]
    mom_y.loc['Flr_4'] = flr_4_wall_demands[4::12]
    mom_y.loc['Flr_5'] = flr_5_wall_demands[4::12]
    mom_y.loc['Flr_6'] = flr_6_wall_demands[4::12]
    mom_y.loc['Flr_7'] = flr_7_wall_demands[4::12]
    mom_y.loc['Flr_8'] = flr_8_wall_demands[4::12]
    mom_y.loc['Flr_9'] = flr_9_wall_demands[4::12]
    mom_y.loc['Flr_10'] = flr_10_wall_demands[4::12]
    mom_y.loc['Flr_11'] = flr_11_wall_demands[4::12]

    mom_z.loc['Flr_1'] = flr_1_wall_demands[5::12]
    mom_z.loc['Flr_2'] = flr_2_wall_demands[5::12]
    mom_z.loc['Flr_3'] = flr_3_wall_demands[5::12]
    mom_z.loc['Flr_4'] = flr_4_wall_demands[5::12]
    mom_z.loc['Flr_5'] = flr_5_wall_demands[5::12]
    mom_z.loc['Flr_6'] = flr_6_wall_demands[5::12]
    mom_z.loc['Flr_7'] = flr_7_wall_demands[5::12]
    mom_z.loc['Flr_8'] = flr_8_wall_demands[5::12]
    mom_z.loc['Flr_9'] = flr_9_wall_demands[5::12]
    mom_z.loc['Flr_10'] = flr_10_wall_demands[5::12]
    mom_z.loc['Flr_11'] = flr_11_wall_demands[5::12]

    return force_x, force_y, force_z, mom_x, mom_y, mom_z


def get_mrsa_wall_rigid_links_demands(modal_combo, results_directory, angular_freq, damping_ratio, num_modes):

    flr_1_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor01_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_2_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor02_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_3_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor03_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_4_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor04_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_5_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor05_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_6_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor06_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_7_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor07_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_8_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor08_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_9_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor09_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_10_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor10_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)
    flr_11_rigidLinks_demands = modal_combo(np.loadtxt(results_directory + 'floor11_wallRigidLinkResp.txt'), angular_freq, damping_ratio, num_modes)

    force_x = pd.DataFrame(columns=['Wall_1L', 'Wall_1R', 'Wall_2L', 'Wall_2R', 'Wall_3L', 'Wall_3R', 'Wall_4L', 'Wall_4R',
                                    'Wall_5L', 'Wall_5R', 'Wall_6L', 'Wall_6R', 'Wall_7L', 'Wall_7R', 'Wall_8L', 'Wall_8R',
                                    'Wall_9L', 'Wall_9R', 'Wall_10L', 'Wall_10R'],
                           index=['Flr_1', 'Flr_2', 'Flr_3', 'Flr_4', 'Flr_5',
                                  'Flr_6', 'Flr_7', 'Flr_8', 'Flr_9', 'Flr_10', 'Flr_11'])

    force_y = pd.DataFrame(columns=force_x.columns, index=force_x.index)
    force_z = pd.DataFrame(columns=force_x.columns, index=force_x.index)
    mom_x = pd.DataFrame(columns=force_x.columns, index=force_x.index)
    mom_y = pd.DataFrame(columns=force_x.columns, index=force_x.index)
    mom_z = pd.DataFrame(columns=force_x.columns, index=force_x.index)

    force_x.loc['Flr_1'] = flr_1_rigidLinks_demands[0::12]
    force_x.loc['Flr_2'] = flr_2_rigidLinks_demands[0::12]
    force_x.loc['Flr_3'] = flr_3_rigidLinks_demands[0::12]
    force_x.loc['Flr_4'] = flr_4_rigidLinks_demands[0::12]
    force_x.loc['Flr_5'] = flr_5_rigidLinks_demands[0::12]
    force_x.loc['Flr_6'] = flr_6_rigidLinks_demands[0::12]
    force_x.loc['Flr_7'] = flr_7_rigidLinks_demands[0::12]
    force_x.loc['Flr_8'] = flr_8_rigidLinks_demands[0::12]
    force_x.loc['Flr_9'] = flr_9_rigidLinks_demands[0::12]
    force_x.loc['Flr_10'] = flr_10_rigidLinks_demands[0::12]
    force_x.loc['Flr_11'] = flr_11_rigidLinks_demands[0::12]

    force_y.loc['Flr_1'] = flr_1_rigidLinks_demands[1::12]
    force_y.loc['Flr_2'] = flr_2_rigidLinks_demands[1::12]
    force_y.loc['Flr_3'] = flr_3_rigidLinks_demands[1::12]
    force_y.loc['Flr_4'] = flr_4_rigidLinks_demands[1::12]
    force_y.loc['Flr_5'] = flr_5_rigidLinks_demands[1::12]
    force_y.loc['Flr_6'] = flr_6_rigidLinks_demands[1::12]
    force_y.loc['Flr_7'] = flr_7_rigidLinks_demands[1::12]
    force_y.loc['Flr_8'] = flr_8_rigidLinks_demands[1::12]
    force_y.loc['Flr_9'] = flr_9_rigidLinks_demands[1::12]
    force_y.loc['Flr_10'] = flr_10_rigidLinks_demands[1::12]
    force_y.loc['Flr_11'] = flr_11_rigidLinks_demands[1::12]

    force_z.loc['Flr_1'] = flr_1_rigidLinks_demands[2::12]
    force_z.loc['Flr_2'] = flr_2_rigidLinks_demands[2::12]
    force_z.loc['Flr_3'] = flr_3_rigidLinks_demands[2::12]
    force_z.loc['Flr_4'] = flr_4_rigidLinks_demands[2::12]
    force_z.loc['Flr_5'] = flr_5_rigidLinks_demands[2::12]
    force_z.loc['Flr_6'] = flr_6_rigidLinks_demands[2::12]
    force_z.loc['Flr_7'] = flr_7_rigidLinks_demands[2::12]
    force_z.loc['Flr_8'] = flr_8_rigidLinks_demands[2::12]
    force_z.loc['Flr_9'] = flr_9_rigidLinks_demands[2::12]
    force_z.loc['Flr_10'] = flr_10_rigidLinks_demands[2::12]
    force_z.loc['Flr_11'] = flr_11_rigidLinks_demands[2::12]

    mom_x.loc['Flr_1'] = flr_1_rigidLinks_demands[3::12]
    mom_x.loc['Flr_2'] = flr_2_rigidLinks_demands[3::12]
    mom_x.loc['Flr_3'] = flr_3_rigidLinks_demands[3::12]
    mom_x.loc['Flr_4'] = flr_4_rigidLinks_demands[3::12]
    mom_x.loc['Flr_5'] = flr_5_rigidLinks_demands[3::12]
    mom_x.loc['Flr_6'] = flr_6_rigidLinks_demands[3::12]
    mom_x.loc['Flr_7'] = flr_7_rigidLinks_demands[3::12]
    mom_x.loc['Flr_8'] = flr_8_rigidLinks_demands[3::12]
    mom_x.loc['Flr_9'] = flr_9_rigidLinks_demands[3::12]
    mom_x.loc['Flr_10'] = flr_10_rigidLinks_demands[3::12]
    mom_x.loc['Flr_11'] = flr_11_rigidLinks_demands[3::12]

    mom_y.loc['Flr_1'] = flr_1_rigidLinks_demands[4::12]
    mom_y.loc['Flr_2'] = flr_2_rigidLinks_demands[4::12]
    mom_y.loc['Flr_3'] = flr_3_rigidLinks_demands[4::12]
    mom_y.loc['Flr_4'] = flr_4_rigidLinks_demands[4::12]
    mom_y.loc['Flr_5'] = flr_5_rigidLinks_demands[4::12]
    mom_y.loc['Flr_6'] = flr_6_rigidLinks_demands[4::12]
    mom_y.loc['Flr_7'] = flr_7_rigidLinks_demands[4::12]
    mom_y.loc['Flr_8'] = flr_8_rigidLinks_demands[4::12]
    mom_y.loc['Flr_9'] = flr_9_rigidLinks_demands[4::12]
    mom_y.loc['Flr_10'] = flr_10_rigidLinks_demands[4::12]
    mom_y.loc['Flr_11'] = flr_11_rigidLinks_demands[4::12]

    mom_z.loc['Flr_1'] = flr_1_rigidLinks_demands[5::12]
    mom_z.loc['Flr_2'] = flr_2_rigidLinks_demands[5::12]
    mom_z.loc['Flr_3'] = flr_3_rigidLinks_demands[5::12]
    mom_z.loc['Flr_4'] = flr_4_rigidLinks_demands[5::12]
    mom_z.loc['Flr_5'] = flr_5_rigidLinks_demands[5::12]
    mom_z.loc['Flr_6'] = flr_6_rigidLinks_demands[5::12]
    mom_z.loc['Flr_7'] = flr_7_rigidLinks_demands[5::12]
    mom_z.loc['Flr_8'] = flr_8_rigidLinks_demands[5::12]
    mom_z.loc['Flr_9'] = flr_9_rigidLinks_demands[5::12]
    mom_z.loc['Flr_10'] = flr_10_rigidLinks_demands[5::12]
    mom_z.loc['Flr_11'] = flr_11_rigidLinks_demands[5::12]

    return force_x, force_y, force_z, mom_x, mom_y, mom_z
