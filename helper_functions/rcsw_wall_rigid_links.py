# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:46:42 2023

@author: Uzo Uwaoma - udu@uw.edu
"""


def create_wall_rigid_links(ops, floor_num, wall_ends_node_tags, lfre_node_tags, wall_link_prop):

    wall_rigid_tag = int('5' + floor_num + '01') # 50101

    wall_link_A = wall_link_prop[0]
    wall_link_E = wall_link_prop[1]
    wall_link_G = wall_link_prop[2]
    wall_link_J = wall_link_prop[3]
    wall_link_I = wall_link_prop[4]
    wall_link_transf_tag_x = wall_link_prop[5]
    wall_link_transf_tag_y = wall_link_prop[6]

    # Wall 1
    ops.element('elasticBeamColumn', wall_rigid_tag, wall_ends_node_tags.loc['wall1_l'][floor_num],
                lfre_node_tags.loc['wall1'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_y)

    ops.element('elasticBeamColumn', wall_rigid_tag + 1, lfre_node_tags.loc['wall1'][floor_num],
                wall_ends_node_tags.loc['wall1_r'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_y)

    # Wall 2
    ops.element('elasticBeamColumn', wall_rigid_tag + 2, wall_ends_node_tags.loc['wall2_l'][floor_num],
                lfre_node_tags.loc['wall2'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_y)

    ops.element('elasticBeamColumn', wall_rigid_tag + 3, lfre_node_tags.loc['wall2'][floor_num],
                wall_ends_node_tags.loc['wall2_r'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_y)

    # Wall 3
    ops.element('elasticBeamColumn', wall_rigid_tag + 4, wall_ends_node_tags.loc['wall3_l'][floor_num],
                lfre_node_tags.loc['wall3'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_x)

    ops.element('elasticBeamColumn', wall_rigid_tag + 5, lfre_node_tags.loc['wall3'][floor_num],
                wall_ends_node_tags.loc['wall3_r'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_x)

    # Wall 4
    ops.element('elasticBeamColumn', wall_rigid_tag + 6, wall_ends_node_tags.loc['wall4_l'][floor_num],
                lfre_node_tags.loc['wall4'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_x)

    ops.element('elasticBeamColumn', wall_rigid_tag + 7, lfre_node_tags.loc['wall4'][floor_num],
                wall_ends_node_tags.loc['wall4_r'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_x)

    # Wall 5
    ops.element('elasticBeamColumn', wall_rigid_tag + 8, wall_ends_node_tags.loc['wall5_l'][floor_num],
                lfre_node_tags.loc['wall5'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_y)

    ops.element('elasticBeamColumn', wall_rigid_tag + 9, lfre_node_tags.loc['wall5'][floor_num],
                wall_ends_node_tags.loc['wall5_r'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_y)

    # Wall 6
    ops.element('elasticBeamColumn', wall_rigid_tag + 10, wall_ends_node_tags.loc['wall6_l'][floor_num],
                lfre_node_tags.loc['wall6'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_y)

    ops.element('elasticBeamColumn', wall_rigid_tag + 11, lfre_node_tags.loc['wall6'][floor_num],
                wall_ends_node_tags.loc['wall6_r'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_y)

    # Wall 7
    ops.element('elasticBeamColumn', wall_rigid_tag + 12, wall_ends_node_tags.loc['wall7_l'][floor_num],
                lfre_node_tags.loc['wall7'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_y)

    ops.element('elasticBeamColumn', wall_rigid_tag + 13, lfre_node_tags.loc['wall7'][floor_num],
                wall_ends_node_tags.loc['wall7_r'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_y)

    # Wall 8
    ops.element('elasticBeamColumn', wall_rigid_tag + 14, wall_ends_node_tags.loc['wall8_l'][floor_num],
                lfre_node_tags.loc['wall8'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_x)

    ops.element('elasticBeamColumn', wall_rigid_tag + 15, lfre_node_tags.loc['wall8'][floor_num],
                wall_ends_node_tags.loc['wall8_r'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_x)

    # Wall 9
    ops.element('elasticBeamColumn', wall_rigid_tag + 16, wall_ends_node_tags.loc['wall9_l'][floor_num],
                lfre_node_tags.loc['wall9'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_x)

    ops.element('elasticBeamColumn', wall_rigid_tag + 17, lfre_node_tags.loc['wall9'][floor_num],
                wall_ends_node_tags.loc['wall9_r'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_x)

    # Wall 10
    ops.element('elasticBeamColumn', wall_rigid_tag + 18, wall_ends_node_tags.loc['wall10_l'][floor_num],
                lfre_node_tags.loc['wall10'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_x)

    ops.element('elasticBeamColumn', wall_rigid_tag + 19, lfre_node_tags.loc['wall10'][floor_num],
                wall_ends_node_tags.loc['wall10_r'][floor_num], wall_link_A, wall_link_E, wall_link_G,
                wall_link_J, wall_link_I, wall_link_I, wall_link_transf_tag_x)
