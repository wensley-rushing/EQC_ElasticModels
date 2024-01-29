# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:14:04 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import os
import sys

sys.path.append('../')

from helper_functions.set_recorders import create_beam_recorders, create_column_recorders, create_wall_recorders, create_wall_rigid_link_recorders

def perform_mrsa(ops, spect_acc, spect_periods, num_modes, results_direc,
                 lfre_node_tags, not_for_optimization, com_node_tags=None, lfrs=None,
                 wall_ends_node_tags=None):

    # print("COM node tags: ", list(com_node_tags.values()))
    direcs = [1, 2]  # Directions for MRSA
    axis = ['X', 'Y']

    # Maintain constant gravity loads and reset time to zero
    ops.loadConst('-time', 0.0)

    for ii in range (len(direcs)):

        # Create directory to save results
        mrsa_res_folder = results_direc + axis[ii] + '/'
        os.makedirs(mrsa_res_folder, exist_ok=True)

        if not_for_optimization:

            # Create recorders for column response in direction of excitation
            create_column_recorders(ops, mrsa_res_folder)

            if lfrs == 'ssmf':

                # Create recorders for beam response in direction of excitation
                create_beam_recorders(ops, mrsa_res_folder)

                # Create recorders to store nodal displacements at the building edges
                ops.recorder('Node', '-file', mrsa_res_folder + 'lowerLeftCornerDisp.txt',
                             '-node', *list(lfre_node_tags.loc['col1'])[1:], '-dof', direcs[ii], 'disp')

                ops.recorder('Node', '-file', mrsa_res_folder + 'lowerRightCornerDisp.txt',
                             '-node', *list(lfre_node_tags.loc['col5'])[1:], '-dof', direcs[ii], 'disp')

                ops.recorder('Node', '-file', mrsa_res_folder + 'upperRightCornerDisp.txt',
                             '-node', *list(lfre_node_tags.loc['col23'])[1:], '-dof', direcs[ii], 'disp')


            if lfrs == 'rcsw':

                # Create recorders for wall response in direction of excitation
                create_wall_recorders(ops, mrsa_res_folder)

                # Create recorders for wall rigid links in direction of excitation
                create_wall_rigid_link_recorders(ops, mrsa_res_folder)

                # Create recorders to store nodal displacements at the building edges
                ops.recorder('Node', '-file', mrsa_res_folder + 'lowerLeftCornerDisp.txt',
                              '-node', *list(wall_ends_node_tags.loc['wall1_l'])[1:], '-dof', direcs[ii], 'disp')

                ops.recorder('Node', '-file', mrsa_res_folder + 'upperRightCornerDisp.txt',
                              '-node', *list(wall_ends_node_tags.loc['wall10_r'])[1:], '-dof', direcs[ii], 'disp')

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



# ops.recorder('Element', '-file', mrsa_res_folder + 'floor01_colResp.txt', '-precision', 9, '-region', 301, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor02_colResp.txt', '-precision', 9, '-region', 302, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor03_colResp.txt', '-precision', 9, '-region', 303, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor04_colResp.txt', '-precision', 9, '-region', 304, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor05_colResp.txt', '-precision', 9, '-region', 305, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor06_colResp.txt', '-precision', 9, '-region', 306, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor07_colResp.txt', '-precision', 9, '-region', 307, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor08_colResp.txt', '-precision', 9, '-region', 308, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor09_colResp.txt', '-precision', 9, '-region', 309, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor10_colResp.txt', '-precision', 9, '-region', 310, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor11_colResp.txt', '-precision', 9, '-region', 311, 'force')

# # Create recorders for wall response in direction of excitation
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor01_wallResp.txt', '-precision', 9, '-region', 401, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor02_wallResp.txt', '-precision', 9, '-region', 402, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor03_wallResp.txt', '-precision', 9, '-region', 403, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor04_wallResp.txt', '-precision', 9, '-region', 404, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor05_wallResp.txt', '-precision', 9, '-region', 405, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor06_wallResp.txt', '-precision', 9, '-region', 406, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor07_wallResp.txt', '-precision', 9, '-region', 407, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor08_wallResp.txt', '-precision', 9, '-region', 408, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor09_wallResp.txt', '-precision', 9, '-region', 409, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor10_wallResp.txt', '-precision', 9, '-region', 410, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor11_wallResp.txt', '-precision', 9, '-region', 411, 'force')

# Create recorders for wall rigid links in direction of excitation
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor01_wallRigidLinkResp.txt', '-precision', 9, '-region', 501, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor02_wallRigidLinkResp.txt', '-precision', 9, '-region', 502, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor03_wallRigidLinkResp.txt', '-precision', 9, '-region', 503, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor04_wallRigidLinkResp.txt', '-precision', 9, '-region', 504, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor05_wallRigidLinkResp.txt', '-precision', 9, '-region', 505, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor06_wallRigidLinkResp.txt', '-precision', 9, '-region', 506, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor07_wallRigidLinkResp.txt', '-precision', 9, '-region', 507, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor08_wallRigidLinkResp.txt', '-precision', 9, '-region', 508, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor09_wallRigidLinkResp.txt', '-precision', 9, '-region', 509, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor10_wallRigidLinkResp.txt', '-precision', 9, '-region', 510, 'force')
# ops.recorder('Element', '-file', mrsa_res_folder + 'floor11_wallRigidLinkResp.txt', '-precision', 9, '-region', 511, 'force')

# ops.recorder('Element', '-file', mrsa_res_folder + 'floor01_wallRigid2elem.txt', '-precision', 9, '-ele', 50101, 50102, 'force')
