# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:45:37 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import numpy as np
import pandas as pd


def get_mvlem_element_demands(wall_demands):

    """
    Helper function to compute the element forces of MVLEM walls.

    There are 10 wall elements on each floor with each wall element defined by
    4 nodes (i, j, k & l.).

    This model has been built such that the i, j, k, & l nodes for each wall
    element are as depicted below (see Fig. 2b in
                                   https://www.sciencedirect.com/science/article/pii/S0141029621002558).

     l--------------------k
     -                    -
     -                    -
     -                    -
     -                    -
     i--------------------j

    At each node, there are 6-DOFs, hence each wall element has 24-DOFs.
    [Fxi, Fyi, Fzi, Mxi, Myi, Mzi, Fxj, Fyj, Fzj, Mxj, Myj, Mzj,
     Fxk, Fyk, Fzk, Mxk, Myk, Mzk, Fxl, Fyl, Fzl, Mxl, Myl, Mzl]

    Total DOFs per floor = 10 * 24 = 240.

    Parameters
    ----------
    wall_demands : numpy.array
        Contains the nodal element forces for a MVLEM-3D Quad element.
    """

    # For each node in the MVLEM Quad element, extract the 6 nodal forces
    nodal_Fx = wall_demands[0::6]
    nodal_Fy = wall_demands[1::6]
    nodal_Fz = wall_demands[2::6]
    nodal_Mx = wall_demands[3::6]
    nodal_My = wall_demands[4::6]
    nodal_Mz = wall_demands[5::6]

    '''
    The sum of the nodal forces (Fx, Fy, Fz, Mx, My or Mz)
    at the top 2 nodes (k & l) of each MVLEM element node is equal
    to the sum at the bottom 2 nodes (i & j).

    Hence, we take the sum of the demands at the top 2 nodes, and that of the
    bottom 2 nodes, then extract only the demands at the bottom nodes.

    '''

    wall_Fx = (nodal_Fx[0::2] + nodal_Fx[1::2])[0::2]
    wall_Fy = (nodal_Fy[0::2] + nodal_Fy[1::2])[0::2]
    wall_Fz = (nodal_Fz[0::2] + nodal_Fz[1::2])[0::2]
    wall_Mx = (nodal_Mx[0::2] + nodal_Mx[1::2])[0::2]
    wall_My = (nodal_My[0::2] + nodal_My[1::2])[0::2]
    wall_Mz = (nodal_Mz[0::2] + nodal_Mz[1::2])[0::2]

    return wall_Fx, wall_Fy, wall_Fz, wall_Mx, wall_My, wall_Mz


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

    (force_x.loc['Flr_1'], force_y.loc['Flr_1'], force_z.loc['Flr_1'],
     mom_x.loc['Flr_1'], mom_y.loc['Flr_1'], mom_z.loc['Flr_1']) = get_mvlem_element_demands(flr_1_wall_demands)

    (force_x.loc['Flr_2'], force_y.loc['Flr_2'], force_z.loc['Flr_2'],
     mom_x.loc['Flr_2'], mom_y.loc['Flr_2'], mom_z.loc['Flr_2']) = get_mvlem_element_demands(flr_2_wall_demands)

    (force_x.loc['Flr_3'], force_y.loc['Flr_3'], force_z.loc['Flr_3'],
     mom_x.loc['Flr_3'], mom_y.loc['Flr_3'], mom_z.loc['Flr_3']) = get_mvlem_element_demands(flr_3_wall_demands)

    (force_x.loc['Flr_4'], force_y.loc['Flr_4'], force_z.loc['Flr_4'],
     mom_x.loc['Flr_4'], mom_y.loc['Flr_4'], mom_z.loc['Flr_4']) = get_mvlem_element_demands(flr_4_wall_demands)

    (force_x.loc['Flr_5'], force_y.loc['Flr_5'], force_z.loc['Flr_5'],
     mom_x.loc['Flr_5'], mom_y.loc['Flr_5'], mom_z.loc['Flr_5']) = get_mvlem_element_demands(flr_5_wall_demands)

    (force_x.loc['Flr_6'], force_y.loc['Flr_6'], force_z.loc['Flr_6'],
     mom_x.loc['Flr_6'], mom_y.loc['Flr_6'], mom_z.loc['Flr_6']) = get_mvlem_element_demands(flr_6_wall_demands)

    (force_x.loc['Flr_7'], force_y.loc['Flr_7'], force_z.loc['Flr_7'],
     mom_x.loc['Flr_7'], mom_y.loc['Flr_7'], mom_z.loc['Flr_7']) = get_mvlem_element_demands(flr_7_wall_demands)

    (force_x.loc['Flr_8'], force_y.loc['Flr_8'], force_z.loc['Flr_8'],
     mom_x.loc['Flr_8'], mom_y.loc['Flr_8'], mom_z.loc['Flr_8']) = get_mvlem_element_demands(flr_8_wall_demands)

    (force_x.loc['Flr_9'], force_y.loc['Flr_9'], force_z.loc['Flr_9'],
     mom_x.loc['Flr_9'], mom_y.loc['Flr_9'], mom_z.loc['Flr_9']) = get_mvlem_element_demands(flr_9_wall_demands)

    (force_x.loc['Flr_10'], force_y.loc['Flr_10'], force_z.loc['Flr_10'],
     mom_x.loc['Flr_10'], mom_y.loc['Flr_10'], mom_z.loc['Flr_10']) = get_mvlem_element_demands(flr_10_wall_demands)

    (force_x.loc['Flr_11'], force_y.loc['Flr_11'], force_z.loc['Flr_11'],
     mom_x.loc['Flr_11'], mom_y.loc['Flr_11'], mom_z.loc['Flr_11']) = get_mvlem_element_demands(flr_11_wall_demands)

    return force_x, force_y, force_z, mom_x, mom_y, mom_z
