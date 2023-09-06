# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:20:32 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import openseespy.opensees as ops


# Define Units
sec = 1

# US units
inch = 1
kips = 1

ft = 12 * inch
lb = kips/1000
ksi = kips/inch**2
psi = ksi/1000
grav_US = 386.4 * inch/sec**2

# Metric Units
m = 1
kN  = 1

mm = m / 1000
N = kN/1000
kPa = 0.001 * N/mm**2   # Kilopascal
MPa = 1 * N/mm**2       # Megapascal
GPa = 1000 * N/mm**2    # Gigapascal
grav_metric = 9.81 * m/sec**2

steel_E = 210 * GPa


def create_beams(floor_num, elev, com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop):
    bm_tag = int('3' + floor_num + '01')   # 30101

    create_bm_joint_offset(bm_tag, 'col2',  'col3', floor_num, elev, 'EW', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 1
    create_bm_joint_offset(bm_tag + 1, 'col3',  'col4', floor_num, elev, 'EW', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 2
    create_bm_joint_offset(bm_tag + 2, 'col4',  'col5', floor_num, elev, 'EW', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 3
    create_bm_joint_offset(bm_tag + 3, 'col14',  'col15', floor_num, elev, 'EW', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 4
    create_bm_joint_offset(bm_tag + 4, 'col21',  'col22', floor_num, elev, 'EW', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 5
    create_bm_joint_offset(bm_tag + 5, 'col22',  'col23', floor_num, elev, 'EW', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 6

    create_bm_joint_offset(bm_tag + 6, 'col1',  'col8', floor_num, elev, 'NS', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)   # Beam 7
    create_bm_joint_offset(bm_tag + 7, 'col8',  'col13', floor_num, elev, 'NS', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 8
    create_bm_joint_offset(bm_tag + 8, 'col11',  'col16', floor_num, elev, 'NS', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 9
    create_bm_joint_offset(bm_tag + 9, 'col16',  'col19', floor_num, elev, 'NS', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 10
    create_bm_joint_offset(bm_tag + 10, 'col7',  'col12', floor_num, elev, 'NS', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 11
    create_bm_joint_offset(bm_tag + 11, 'col12',  'col17', floor_num, elev, 'NS', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 12
    create_bm_joint_offset(bm_tag + 12, 'col17',  'col20', floor_num, elev, 'NS', com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop)  # Beam 13


# ============================================================================
# Create beam elements and rigid panel zone beam elements
# ============================================================================
def create_bm_joint_offset(bm_tag, left_col, right_col, floor_num, elev, bm_orient, com_node, smf_node_tags, smf_coords_df, bm_prop, col_prop):

    # ==========================================================
    # Extract beam section properties
    # ==========================================================

    bm_section_prop = bm_prop[0]
    bm_E = bm_prop[1]
    bm_G = bm_prop[2]
    bm_transf_tag_x = bm_prop[3]
    bm_transf_tag_y = bm_prop[4]

    bm_A = bm_section_prop['Ag'] * mm**2
    bm_Iy = bm_section_prop['Iy'] * 1E6 * mm**4   # weak axis
    bm_Iz = bm_section_prop['Ix']  * 1E6 * mm**4   # strong axis
    bm_J = bm_section_prop['J'] * 1E3 * mm**4


    # ==========================================================
    # Extract properties of panel zone rigid beam elements
    # ==========================================================
    pz_w = col_prop['d.1'] * mm / 2  # Half the depth of panel zone region
    pz_A = col_prop['A.1'] * 100 * mm**2
    pz_E = bm_E * 100
    pz_G = bm_G
    pz_J = col_prop['J.1'] * 1E3 * mm**4
    pz_I = col_prop['Ix.1'] * 1E6 * mm**4

    pzone_transf_tag_bm_x = bm_prop[5]
    pzone_transf_tag_bm_y = bm_prop[6]

    # ==========================================================
    # Create additional nodes for rigid panel zone beam elemnts.
    # Define beam and panel zone elements.
    # ==========================================================

    '''
    There are additional nodes provided to the right and left of each beam-column joint
    to account for rigid panel zone beam elements.

    Nodes to the left and right of the beam-column joint are tagged relative to the beam-column joint
    node.

    A '3' is appended to the beam-column joint node for the node to the left of the joint.
    A '4' is appended to the beam-column joint node for the node to the right of the joint.
    '''

    left_col_node = smf_node_tags.loc[left_col][floor_num]
    right_col_node = smf_node_tags.loc[right_col][floor_num]

    left_rigid_link_j_node = int(str(left_col_node) + '4')
    right_rigid_link_i_node = int(str(right_col_node) + '3')

    if bm_orient == 'EW':

        ops.node(left_rigid_link_j_node, smf_coords_df.loc[left_col]['x'] + pz_w, smf_coords_df.loc[left_col]['y'], elev)
        ops.node(right_rigid_link_i_node, smf_coords_df.loc[right_col]['x'] - pz_w, smf_coords_df.loc[right_col]['y'], elev)

        # Impose rigid diaphragm constraint
        ops.rigidDiaphragm(3, com_node, *[left_rigid_link_j_node, right_rigid_link_i_node])

        ops.element('elasticBeamColumn', int(str(bm_tag) + '3'), left_col_node,
                    left_rigid_link_j_node, pz_A, pz_E, pz_G, pz_J, pz_I, pz_I, pzone_transf_tag_bm_x) # Rigid link to the left of beam

        ops.element('elasticBeamColumn', bm_tag, left_rigid_link_j_node,
                    right_rigid_link_i_node, bm_A, bm_E, bm_G, bm_J, bm_Iy,
                    bm_Iz, bm_transf_tag_x)  # Beam

        ops.element('elasticBeamColumn', int(str(bm_tag) + '4'), right_rigid_link_i_node,
                    right_col_node, pz_A, pz_E, pz_G, pz_J, pz_I, pz_I, pzone_transf_tag_bm_x) # Rigid link to the right of beam

    else: # bm_orient = NS

        ops.node(left_rigid_link_j_node, smf_coords_df.loc[left_col]['x'], smf_coords_df.loc[left_col]['y'] + pz_w, elev)
        ops.node(right_rigid_link_i_node, smf_coords_df.loc[right_col]['x'], smf_coords_df.loc[right_col]['y'] - pz_w, elev)

        # Impose rigid diaphragm constraint
        ops.rigidDiaphragm(3, com_node, *[left_rigid_link_j_node, right_rigid_link_i_node])

        ops.element('elasticBeamColumn', int(str(bm_tag) + '3'), left_col_node,
                    left_rigid_link_j_node, pz_A, pz_E, pz_G, pz_J, pz_I, pz_I, pzone_transf_tag_bm_y)

        ops.element('elasticBeamColumn', bm_tag, left_rigid_link_j_node,
                    right_rigid_link_i_node, bm_A, bm_E, bm_G, bm_J, bm_Iy,
                    bm_Iz, bm_transf_tag_y)  # Beam

        ops.element('elasticBeamColumn', int(str(bm_tag) + '4'), right_rigid_link_i_node,
                    right_col_node, pz_A, pz_E, pz_G, pz_J, pz_I, pz_I, pzone_transf_tag_bm_y)

