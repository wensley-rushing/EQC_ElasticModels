# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:14:04 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import os


def perform_rcsw_mrsa(ops, spect_acc, spect_periods, num_modes, results_direc,
                 lfre_node_tags, com_node_tags=None):

    direcs = [1, 2]  # Directions for MRSA
    axis = ['X', 'Y']

    # Maintain constant gravity loads and reset time to zero
    ops.loadConst('-time', 0.0)

    for ii in range (len(direcs)):

        # Create directory to save results
        mrsa_res_folder = results_direc + axis[ii] + '/'
        os.makedirs(mrsa_res_folder, exist_ok=True)

        # Create recorders for column response in direction of excitation
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor01_colResp.txt', '-precision', 9, '-region', 301, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor02_colResp.txt', '-precision', 9, '-region', 302, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor03_colResp.txt', '-precision', 9, '-region', 303, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor04_colResp.txt', '-precision', 9, '-region', 304, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor05_colResp.txt', '-precision', 9, '-region', 305, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor06_colResp.txt', '-precision', 9, '-region', 306, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor07_colResp.txt', '-precision', 9, '-region', 307, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor08_colResp.txt', '-precision', 9, '-region', 308, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor09_colResp.txt', '-precision', 9, '-region', 309, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor10_colResp.txt', '-precision', 9, '-region', 310, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor11_colResp.txt', '-precision', 9, '-region', 311, 'force')

        # Create recorders for wall response in direction of excitation
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor01_wallResp.txt', '-precision', 9, '-region', 401, 'globalForce')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor02_wallResp.txt', '-precision', 9, '-region', 402, 'globalForce')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor03_wallResp.txt', '-precision', 9, '-region', 403, 'globalForce')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor04_wallResp.txt', '-precision', 9, '-region', 404, 'globalForce')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor05_wallResp.txt', '-precision', 9, '-region', 405, 'globalForce')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor06_wallResp.txt', '-precision', 9, '-region', 406, 'globalForce')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor07_wallResp.txt', '-precision', 9, '-region', 407, 'globalForce')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor08_wallResp.txt', '-precision', 9, '-region', 408, 'globalForce')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor09_wallResp.txt', '-precision', 9, '-region', 409, 'globalForce')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor10_wallResp.txt', '-precision', 9, '-region', 410, 'globalForce')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor11_wallResp.txt', '-precision', 9, '-region', 411, 'globalForce')


        # Create recorders to store nodal displacements at the building edges
        # ops.recorder('Node', '-file', mrsa_res_folder + 'lowerLeftCornerDisp.txt',
        #               '-node', *list(wall_ends_node_tags.loc['wall1_l'])[1:], '-dof', direcs[ii], 'disp')

        # ops.recorder('Node', '-file', mrsa_res_folder + 'upperRightCornerDisp.txt',
        #               '-node', *list(wall_ends_node_tags.loc['wall10_r'])[1:], '-dof', direcs[ii], 'disp')

        ops.recorder('Node', '-file', mrsa_res_folder + 'lowerRightCornerDisp.txt',
                      '-node', *list(lfre_node_tags.loc['col3'])[1:], '-dof', direcs[ii], 'disp')

        # Base shear
        ops.recorder('Node', '-file', mrsa_res_folder + 'baseShear' + axis[ii] + '.txt',
                      '-node', *lfre_node_tags['00'].tolist(), '-dof', direcs[ii], 'reaction')

        if com_node_tags:
            # Recorders for COM displacement
            ops.recorder('Node', '-file', mrsa_res_folder + 'COM_disp' + axis[ii] + '.txt',
                          '-node', *list(com_node_tags.values()), '-dof', direcs[ii], 'disp')

        for jj in range(num_modes):
            ops.responseSpectrumAnalysis(direcs[ii], '-Tn', *spect_periods, '-Sa', *spect_acc, '-mode', jj + 1)

        # Shut down recorder for current direction of excitation
        ops.remove('recorders')
