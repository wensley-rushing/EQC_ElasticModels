# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:42:03 2023

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


def create_columns(floor_num, smf_node_tags, col_prop, beam_prop):

    # ================================
    # Extract column properties
    # ================================
    col_section_prop = col_prop[0]
    col_E = col_prop[1]
    col_G = col_prop[2]

    # col_d = col_section_prop['d.1'] * mm
    col_A = col_section_prop['A.1'] * mm**2
    col_J = col_section_prop['J.1'] * 1E3 * mm**4

    # East-West SMF columns
    col_Iy_EW = col_section_prop['Iy.1'] * 1E6 * mm**4  # weak Iyy
    col_Iz_EW = col_section_prop['Ix.1'] * 1E6 * mm**4   # strong Ixx
    col_transf_tag_EW = col_prop[3]

    #North-South SMF columns
    col_Iy_NS = col_section_prop['Ix.1'] * 1E6 * mm**4  # strong Ixx
    col_Iz_NS = col_section_prop['Iy.1'] * 1E6 * mm**4  # weak Iyy
    col_transf_tag_NS = col_prop[4]

    col_tag = int('2' + floor_num + '01')  # 20101


    # ============================================================================
    # Create columns
    # ============================================================================

    '''
    There are additional nodes provided at the top and bottom of each beam-column joint
    to account for rigid panel zone elements.

    Nodes at the top & bottom  of the beam-column joint are tagged relative to the beam-column joint
    node.

    A '1' is appended to the tag of the beam-column joint node for the node above the joint.
    A '2' is appended to the tag of the beam-column joint node for the node below the joint.

    However, at the bottom of the 1st floor, there is no such node offset as there is no panel zone.
    '''

    if floor_num != '01':
        pz_node_tag = '1'
    else:
        pz_node_tag = ''

    ops.element('elasticBeamColumn', col_tag, int(str(smf_node_tags.loc['col1'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col1'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)

    ops.element('elasticBeamColumn', col_tag + 1, int(str(smf_node_tags.loc['col2'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col2'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_EW, col_Iz_EW, col_transf_tag_EW)



    ops.element('elasticBeamColumn', col_tag + 2, int(str(smf_node_tags.loc['col3'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col3'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_EW, col_Iz_EW, col_transf_tag_EW)



    ops.element('elasticBeamColumn', col_tag + 3, int(str(smf_node_tags.loc['col4'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col4'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_EW, col_Iz_EW, col_transf_tag_EW)



    ops.element('elasticBeamColumn', col_tag + 4, int(str(smf_node_tags.loc['col5'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col5'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_EW, col_Iz_EW, col_transf_tag_EW)


    ops.element('elasticBeamColumn', col_tag + 5, int(str(smf_node_tags.loc['col6'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col6'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)

    ops.element('elasticBeamColumn', col_tag + 6, int(str(smf_node_tags.loc['col7'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col7'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)

    ops.element('elasticBeamColumn', col_tag + 7, int(str(smf_node_tags.loc['col8'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col8'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)

    ops.element('elasticBeamColumn', col_tag + 8, int(str(smf_node_tags.loc['col9'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col9'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)

    ops.element('elasticBeamColumn', col_tag + 9, int(str(smf_node_tags.loc['col10'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col10'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)


    ops.element('elasticBeamColumn', col_tag + 10, int(str(smf_node_tags.loc['col11'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col11'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)

    ops.element('elasticBeamColumn', col_tag + 11, int(str(smf_node_tags.loc['col12'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col12'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)


    ops.element('elasticBeamColumn', col_tag + 12, int(str(smf_node_tags.loc['col13'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col13'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)

    ops.element('elasticBeamColumn', col_tag + 13, int(str(smf_node_tags.loc['col14'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col14'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_EW, col_Iz_EW, col_transf_tag_EW)

    ops.element('elasticBeamColumn', col_tag + 14, int(str(smf_node_tags.loc['col15'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col15'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_EW, col_Iz_EW, col_transf_tag_EW)

    ops.element('elasticBeamColumn', col_tag + 15, int(str(smf_node_tags.loc['col16'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col16'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)

    ops.element('elasticBeamColumn', col_tag + 16, int(str(smf_node_tags.loc['col17'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col17'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)


    ops.element('elasticBeamColumn', col_tag + 17, int(str(smf_node_tags.loc['col18'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col18'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)

    ops.element('elasticBeamColumn', col_tag + 18, int(str(smf_node_tags.loc['col19'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col19'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)

    ops.element('elasticBeamColumn', col_tag + 19, int(str(smf_node_tags.loc['col20'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col20'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_NS, col_Iz_NS, col_transf_tag_NS)


    ops.element('elasticBeamColumn', col_tag + 20, int(str(smf_node_tags.loc['col21'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col21'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_EW, col_Iz_EW, col_transf_tag_EW)

    ops.element('elasticBeamColumn', col_tag + 21, int(str(smf_node_tags.loc['col22'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col22'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_EW, col_Iz_EW, col_transf_tag_EW)

    ops.element('elasticBeamColumn', col_tag + 22, int(str(smf_node_tags.loc['col23'][floor_num] - 10000) + pz_node_tag),
                int(str(smf_node_tags.loc['col23'][floor_num]) + '2'),
                col_A, col_E, col_G, col_J, col_Iy_EW, col_Iz_EW, col_transf_tag_EW)


    # ============================================================================
    # Create rigid panel zone column elements
    # ============================================================================

    pz_A = col_A * 100
    pz_E = col_E * 100
    pz_G = col_G
    pz_J = col_J
    pz_I = col_Iy_NS
    pzone_transf_tag_col = col_prop[5]


    col_tag = int('2' + floor_num + '01')  # 20101
    pz_tag = int(str(col_tag) + '1')  # 201011

    for col in smf_node_tags.index.tolist():
        ops.element('elasticBeamColumn', pz_tag, int(str(smf_node_tags.loc[col][floor_num]) + '2'), smf_node_tags.loc[col][floor_num], pz_A, pz_E, pz_G, pz_J, pz_I, pz_I, pzone_transf_tag_col)

        if floor_num != '11':
            ops.element('elasticBeamColumn', pz_tag + 1, smf_node_tags.loc[col][floor_num], int(str(smf_node_tags.loc[col][floor_num]) + '1'), pz_A, pz_E, pz_G, pz_J, pz_I, pz_I, pzone_transf_tag_col)

        pz_tag += 10

