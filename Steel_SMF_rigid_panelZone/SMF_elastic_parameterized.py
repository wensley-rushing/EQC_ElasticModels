# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:43:33 2023

@author: Uzo Uwaoma
"""
# Import required modules
import os
import sys
import math
import time
import openseespy.opensees as ops
import pandas as pd
import numpy as np

# Append directory of helper functions to Pyhton Path
sys.path.append('../')

from helper_functions.create_floor_shell import refine_mesh
from helper_functions.create_floor_shell import create_shell
from helper_functions.build_smf_column import create_columns
from helper_functions.build_smf_beam import create_beams
from helper_functions.get_beam_demands import process_beam_resp
from helper_functions.get_story_drift import compute_story_drifts
from helper_functions.cqc_modal_combo import modal_combo
from helper_functions.elf_new_zealand import nz_horiz_seismic_shear, nz_horiz_force_distribution


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

print('Basic units are: \n \t Force: kN; \n \t Length: m; \n \t Time: sec.')
print('')

# Floor elevations
typ_flr_height = 3.1 * m
ground_flr = 0.0 * m

flr1 = 4.5 * m
flr2 = flr1 + typ_flr_height
flr3 = flr2 + typ_flr_height
flr4 = flr3 + typ_flr_height
flr5 = flr4 + typ_flr_height
flr6 = flr5 + typ_flr_height
flr7 = flr6 + typ_flr_height
flr8 = flr7 + typ_flr_height
flr9 = flr8 + typ_flr_height
flr10 = flr9 + typ_flr_height
roof_flr = flr10 + typ_flr_height

story_heights = np.array([flr1, typ_flr_height, typ_flr_height, typ_flr_height, typ_flr_height, typ_flr_height,
                 typ_flr_height, typ_flr_height, typ_flr_height, typ_flr_height, typ_flr_height]) # Story heights from Floor 1 to Roof

# Column centerline x-y coordinates in meters
smf_coords_dict = {'col1': [0, 0],
                   'col2': [6.505, 0],
                   'col3': [13.010, 0],
                   'col4': [21.210, 0],
                   'col5': [29.410, 0],
                   'col6': [0., 2.225],
                   'col7': [29.410, 2.225],
                   'col8': [0, 9.425],
                   'col9': [6.505, 9.425],
                   'col10': [13.010, 9.425],
                   'col11': [21.210, 9.425],
                   'col12': [29.410, 9.425],
                   'col13': [0, 16.625],
                   'col14': [6.505, 16.625],
                   'col15': [13.010, 16.625],
                   'col16': [21.210, 16.625],
                   'col17': [29.410, 16.625],
                   'col18': [13.010, 23.825],
                   'col19': [21.210, 23.825],
                   'col20': [29.410, 23.825],
                   'col21': [13.010, 31.025],
                   'col22': [21.210, 31.025],
                   'col23': [29.410, 31.025]}

smf_coords_df = pd.DataFrame.from_dict(smf_coords_dict, orient='index', columns=['x', 'y'])

# Create a dataframe to store node tags nodes at column locations.
smf_node_tags = pd.DataFrame(columns=['00', '01', '02', '03', '04', '05',
                                       '06', '07', '08', '09', '10', '11'],
                             index=smf_coords_df.index)

'Sort x and y-coordinates of SMF. This will be used to define a mesh grid'
# Extract x & y coordinates, sort and remove dupllicates
col_x_coords = sorted(list(set([coord for coord in smf_coords_df['x']])))
col_y_coords = sorted(list(set([coord for coord in smf_coords_df['y']])))

col_x_coords = np.array(list(col_x_coords))
col_y_coords = np.array(list(col_y_coords))

# Create mesh
discretize = 0
if discretize:
    mesh_size = 4 * m  # Mesh size - 4m x 4m elements
    x_coords = refine_mesh(col_x_coords, mesh_size)
    y_coords = refine_mesh(col_y_coords, mesh_size)
else:
    x_coords = col_x_coords
    y_coords = col_y_coords


# Generated mesh grid covers the full rectangular area, trim grid to account for actual building plan
ylim = 16.625  # y-limit of reentrant corner
xlim = 13.010  # x-limit of reentrant corner

mesh_grid_df = pd.DataFrame(columns=['x', 'y'])

row = 0

for x in x_coords:
    for y in y_coords:

        if not((x < xlim) and (y > ylim)):
            mesh_grid_df.loc[row] = [x, y]
            row += 1

mesh_grid_df = mesh_grid_df.round(decimals = 4)

# Extract unique y-coordinate values
unique_ys = mesh_grid_df.y.unique()
num_y_groups = len(unique_ys)

# Group x-coordinates based on y-coordinate value
grouped_x_coord = []

row = 0
for val in unique_ys:
    grouped_x_coord.append(np.array(mesh_grid_df[mesh_grid_df.y == val].x))

# ============================================================================
# Load in AISC steel section database
# ============================================================================
steel_data_metric = pd.read_excel('../../../aisc-shapes-database-v16.0.xlsx', sheet_name='Database v16.0',
                               usecols='A, B, C,CH:CK,CP,CU,CX,DK,DN,DQ:DX,EB,FH', index_col='AISC_Manual_Label')

# Select only W14 sections from database for columns
col_filter = steel_data_metric['EDI_Std_Nomenclature'].str.startswith('W14')

smf_col_database = steel_data_metric[col_filter]

# section_filter = steel_data_metric['Type'] == 'W'
# steel_database_metric = steel_data_metric[section_filter]

# ============================================================================
# Load in New Zealand steel section database
# ============================================================================
steel_data_nzs_UB = pd.read_excel('../../../nzs_steel_sections.xlsx', sheet_name='UB',
                               index_col='Designation')

steel_data_nzs_UC = pd.read_excel('../../../nzs_steel_sections.xlsx', sheet_name='UC',
                               index_col='Designation')
"""
# ============================================================================
# Define shell properties for floor diaphragm
# ============================================================================

nD_mattag = 1
plate_fiber_tag = 2
shell_sect_tag = 1

slab_thick = 165 * mm
fiber_thick = slab_thick / 3

shell_E =  26000 * MPa # Modulus of concrete
shell_nu = 0.2  # Poisson's ratio

# ============================================================================
# Define generic steel properties
# ============================================================================
steel_E = 210 * GPa

# ============================================================================
# Define rigid material for beam-column joints elements in panel zone region
# ============================================================================
pz_E = steel_E
rigid_mat_tag = 100
ops.uniaxialMaterial('Elastic', rigid_mat_tag, pz_E)

pzone_transf_tag_col = 100
pzone_transf_tag_bm_x = 200
pzone_transf_tag_bm_y = 300


def get_plastic_sec_mod(mom_inertiaX):

    plastic_sec_mod = 3.2511 * mom_inertiaX + 492.7

    return plastic_sec_mod


def get_mom_inertiaY(mom_inertiaX):

    mom_inertiaY = 0.0342 * mom_inertiaX + 4.6305

    return mom_inertiaY


def get_polar_mom_inertia(mom_inertiaY):

    polar_mom_inertia = 39.136 * mom_inertiaY - 152.24

    return polar_mom_inertia



# ============================================================================
# Define beam properties
# ============================================================================
bm_nu = 0.28  # Poisson's ratio for steel
bm_E = steel_E
bm_G = bm_E / (2*(1 + bm_nu))

# Initialize array of possible values for beam Ix
bm_mom_inertia_strong = np.array(list(steel_data_nzs_UB['Ix']))

# The geometric properties of the beams will be defined using a 610UB125 (W24x84)
smf_beam_prop = steel_data_nzs_UB.loc['610UB101'].copy()  #  610UB125
# bm_d = smf_beam_prop['d'] * mm

bm_transf_tag_x = 3  # Beams oriented in Global-X direction
bm_transf_tag_y = 4  # Beams oriented in Global-Y direction

base_Ix = 986.00
# bm_Ix_modif = [1, 1, 1, 1, 1]
bm_Ix_modif = [1, 0.8, 0.6, 0.4, 0.2]

bm_sect_flr_1 = steel_data_nzs_UB.loc[steel_data_nzs_UB.index[steel_data_nzs_UB['Ix'] >= bm_Ix_modif[0] * base_Ix].tolist()[-1]]
bm_sect_flr_2_to_4 = steel_data_nzs_UB.loc[steel_data_nzs_UB.index[steel_data_nzs_UB['Ix'] >= bm_Ix_modif[1] * base_Ix].tolist()[-1]]
bm_sect_flr_5_to_7 = steel_data_nzs_UB.loc[steel_data_nzs_UB.index[steel_data_nzs_UB['Ix'] >= bm_Ix_modif[2] * base_Ix].tolist()[-1]]
bm_sect_flr_8_to_10 = steel_data_nzs_UB.loc[steel_data_nzs_UB.index[steel_data_nzs_UB['Ix'] >= bm_Ix_modif[3] * base_Ix].tolist()[-1]]
bm_sect_flr_11 = steel_data_nzs_UB.loc[steel_data_nzs_UB.index[steel_data_nzs_UB['Ix'] >= bm_Ix_modif[4] * base_Ix].tolist()[-1]]

# Assume linear relationship
# Base Ix & slope

bm_prop_flr_1 = [bm_sect_flr_1, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_2 = [bm_sect_flr_2_to_4, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_3 = [bm_sect_flr_2_to_4, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_4 = [bm_sect_flr_2_to_4, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_5 = [bm_sect_flr_5_to_7, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_6 = [bm_sect_flr_5_to_7, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_7 = [bm_sect_flr_5_to_7, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_8 = [bm_sect_flr_8_to_10, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_9 = [bm_sect_flr_8_to_10, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_10 = [bm_sect_flr_8_to_10, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_11 = [bm_sect_flr_11, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]

# ============================================================================
# Define column properties
# ============================================================================
col_nu = 0.28  # Poisson's ratio for steel
col_E = steel_E
col_G = col_E / (2*(1 + col_nu))

# The geometric properties of the columns for the East-West SMF will be defined using a W14x132 (metric W360x196)
smf_col_EW_prop = steel_database_metric.loc['W14X132'].copy()
col_d = smf_col_EW_prop['d.1'] * mm
col_A_EW = smf_col_EW_prop['A.1'] * mm**2
col_Iy_EW = smf_col_EW_prop['Iy.1'] * 1E6 * mm**4   # weak Iyy
col_Iz_EW = smf_col_EW_prop['Ix.1'] * 1E6 * mm**4   # strong Ixx
col_J_EW = smf_col_EW_prop['J.1'] * 1E3 * mm**4

col_transf_tag_EW = 1

col_EW_prop_flr_1 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]
col_EW_prop_flr_2 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]
col_EW_prop_flr_3 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]
col_EW_prop_flr_4 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]
col_EW_prop_flr_5 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]
col_EW_prop_flr_6 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]
col_EW_prop_flr_7 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]
col_EW_prop_flr_8 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]
col_EW_prop_flr_9 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]
col_EW_prop_flr_10 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]
col_EW_prop_flr_11 = [col_d, col_A_EW, col_G, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]

# The geometric properties of the columns for the North-South SMF will be defined using a W14x132 (metric W360x196)
smf_col_NS_prop = steel_database_metric.loc['W14X132'].copy()
col_A_NS = smf_col_NS_prop['A.1'] * mm**2
col_Iy_NS = smf_col_NS_prop['Ix.1'] * 1E6 * mm**4   # strong Ixx
col_Iz_NS = smf_col_NS_prop['Iy.1'] * 1E6 * mm**4   # weak Iyy
col_J_NS = smf_col_NS_prop['J.1'] * 1E3 * mm**4

col_transf_tag_NS = 2

col_NS_prop_flr_1 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]
col_NS_prop_flr_2 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]
col_NS_prop_flr_3 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]
col_NS_prop_flr_4 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]
col_NS_prop_flr_5 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]
col_NS_prop_flr_6 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]
col_NS_prop_flr_7 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]
col_NS_prop_flr_8 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]
col_NS_prop_flr_9 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]
col_NS_prop_flr_10 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]
col_NS_prop_flr_11 = [col_d, col_A_NS, col_G, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS, pzone_transf_tag_col]


# ============================================================================
# Initialize dictionary to store node tags of COM for all floors
# Initialize dictionary to store total mass of each floor
# ============================================================================
com_node_tags = {}
total_floor_mass = {}

# ============================================================================
# Define function to create a floor
# ============================================================================

def create_floor(elev, floor_num, beam_prop=None, col_prop_EW=None, col_prop_NS=None, floor_label='',):

    node_compile = []  # Store node numbers grouped according to their y-coordinates

    # Initialize node tag for bottom-left node
    node_num = int(floor_num + '1000')

    # Create nodes
    for jj in range(len(unique_ys)):
        x_vals = grouped_x_coord[jj]
        node_list = []

        for x_val in x_vals:

            if floor_num == '00': # Only create bottom floor nodes at the location of columns

                # Check if the current node is at the location of a column
                if (smf_coords_df == [x_val, unique_ys[jj]]).all(1).any():

                    ops.node(node_num, x_val, unique_ys[jj], elev)

                    # Assign Boundary conditions
                    ops.fix(node_num, 1, 1, 1, 1, 1, 1)

            else:
                ops.node(node_num, x_val, unique_ys[jj], elev)

            'Store node tags for nodes at the location of columns'
            # Check if the current node is at the location of a wall or column
            if (smf_coords_df == [x_val, unique_ys[jj]]).all(1).any():

                # Get the row index
                row_id = smf_coords_df[(smf_coords_df['x'] == x_val) & (smf_coords_df['y'] == unique_ys[jj])].index.tolist()[0]

                # Assign node tag to `smf_coords_df`
                smf_node_tags.loc[row_id][floor_num] = node_num

                # Create additional nodes for rigid column elements in panel zone region
                # Along column line
                if floor_num != '00':

                    bm_col_joint_node_top = int(str(node_num) + '1')
                    bm_col_joint_node_bot = int(str(node_num) + '2')

                    pz_d = beam_prop[0]['d'] / 2 * mm # Half the depth of panel zone region
                    ops.node(bm_col_joint_node_bot, x_val, unique_ys[jj], elev - pz_d)

                    if floor_num != '11':  # No panel zone above roof level
                        ops.node(bm_col_joint_node_top, x_val, unique_ys[jj], elev + pz_d)

            # Move to next node
            node_list.append(node_num)
            node_num += 1

        node_compile.append(node_list)

    # Store node tag for COM node
    com_node = node_num # Node tag assigned to center of mass for the current floor.

    # Get all node tags in current floor
    floor_node_tags = [node for node_list in node_compile for node in node_list]

    # ========================================================================
    # Create floor diaphragm - Loads & mass are assigned here
    # Compute center of mass
    # Then create columns and beams
    # ========================================================================
    if floor_num != '00':

        # Create shell - Assign loads & mass
        create_shell(floor_num, node_compile, shell_sect_tag, num_y_groups)

        # Compute center of mass
        floor_node_x_coord = [ops.nodeCoord(node, 1) for node in floor_node_tags]
        floor_node_y_coord = [ops.nodeCoord(node, 2) for node in floor_node_tags]

        floor_node_x_mass = [ops.nodeMass(node, 1) for node in floor_node_tags]
        floor_node_y_mass = [ops.nodeMass(node, 2) for node in floor_node_tags]

        # Store total floor mass
        total_floor_mass[floor_num] = round(sum(floor_node_x_mass), 3)

        # Initialize DataFrame to store nodal data for COM computation
        com_data = pd.DataFrame()

        com_data['NodeTags'] = floor_node_tags
        com_data['xCoord'] = floor_node_x_coord
        com_data['yCoord'] = floor_node_y_coord
        com_data['xMass'] = floor_node_x_mass
        com_data['yMass'] = floor_node_y_mass

        com_data['xMass_xCoord'] = com_data['xMass'] * com_data['xCoord']
        com_data['yMass_yCoord'] = com_data['yMass'] * com_data['yCoord']

        com_x = com_data['xMass_xCoord'].sum() / com_data['xMass'].sum()
        com_y = com_data['yMass_yCoord'].sum() / com_data['yMass'].sum()

        # Create COM node
        ops.node(com_node, com_x, com_y, elev)

        # Impose rigid diaphragm constraint
        ops.rigidDiaphragm(3, com_node, *floor_node_tags)

        # Constraints for Rigid Diaphragm Primary node
        ops.fix(com_node, 0, 0, 1, 1, 1, 0)  # dx, dy, dz, rx, ry, rz

        com_node_tags[floor_num] = com_node

        # Create columns & beams
        create_columns(floor_num, smf_node_tags, col_prop_EW, col_prop_NS, beam_prop)
        create_beams(floor_num, elev, com_node, smf_node_tags, smf_coords_df, beam_prop, col_prop_NS)

    print('Floor ' + floor_num + ' created')


# ============================================================================
# Model builder
# ============================================================================
def build_model():

    ops.wipe()

    # Model Builder
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # Create shell material for floor diaphragm
    ops.nDMaterial('ElasticIsotropic', nD_mattag, shell_E, shell_nu)
    ops.nDMaterial('PlateFiber', plate_fiber_tag, nD_mattag)
    ops.section('LayeredShell', shell_sect_tag, 3, plate_fiber_tag, fiber_thick, plate_fiber_tag, fiber_thick, plate_fiber_tag, fiber_thick)

    # Define geometric transformation for beams
    ops.geomTransf('PDelta', bm_transf_tag_x, 0, -1, 0)
    ops.geomTransf('PDelta', bm_transf_tag_y, 1, 0, 0)  # -1, 0, 0

    # Define geometric transformation for columns
    ops.geomTransf('PDelta', col_transf_tag_EW, 0, 1, 0)
    ops.geomTransf('PDelta', col_transf_tag_NS, 0, 1, 0)

    # Define geometric transformation for rigid panel zone elements
    ops.geomTransf('Linear', pzone_transf_tag_col, 0, 1, 0)
    ops.geomTransf('Linear', pzone_transf_tag_bm_x, 0, -1, 0)
    ops.geomTransf('Linear', pzone_transf_tag_bm_y, 1, 0, 0)

    # Create all floors of building
    print('Now creating SSMF model... \n')
    create_floor(ground_flr, '00')
    create_floor(flr1, '01', bm_prop_flr_1, col_EW_prop_flr_1, col_NS_prop_flr_1, '1st')
    create_floor(flr2, '02', bm_prop_flr_2, col_EW_prop_flr_2, col_NS_prop_flr_2, '2nd')
    create_floor(flr3, '03', bm_prop_flr_3, col_EW_prop_flr_3, col_NS_prop_flr_3, '3rd')
    create_floor(flr4, '04', bm_prop_flr_4, col_EW_prop_flr_4, col_NS_prop_flr_4, '4th')
    create_floor(flr5, '05', bm_prop_flr_5, col_EW_prop_flr_5, col_NS_prop_flr_5, '5th')
    create_floor(flr6, '06', bm_prop_flr_6, col_EW_prop_flr_6, col_NS_prop_flr_6, '6th')
    create_floor(flr7, '07', bm_prop_flr_7, col_EW_prop_flr_7, col_NS_prop_flr_7, '7th')
    create_floor(flr8, '08', bm_prop_flr_8, col_EW_prop_flr_8, col_NS_prop_flr_8, '8th')
    create_floor(flr9, '09', bm_prop_flr_9, col_EW_prop_flr_9, col_NS_prop_flr_9, '9th')
    create_floor(flr10, '10', bm_prop_flr_10, col_EW_prop_flr_10, col_NS_prop_flr_10, '10th')
    create_floor(roof_flr, '11', bm_prop_flr_11, col_EW_prop_flr_11, col_NS_prop_flr_11, 'Roof')


    # ============================================================================
    # Create regions for SMF beams based on floor
    # ============================================================================
    # Get all element tags
    elem_tags = ops.getEleTags()

    floor_nums = ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11']

    beam_tags = []

    for floor in floor_nums:
        floor_bm_tags = []

        for tag in elem_tags:
            # Only select beam elements
            if str(tag).startswith('3' + floor):
                floor_bm_tags.append(tag)

        beam_tags.append(floor_bm_tags)

    ops.region(301, '-eleOnly', *beam_tags[0])  # Region for all beams on 1st floor
    ops.region(302, '-eleOnly', *beam_tags[1])  # Region for all beams on 2nd floor
    ops.region(303, '-eleOnly', *beam_tags[2])  # Region for all beams on 3rd floor
    ops.region(304, '-eleOnly', *beam_tags[3])  # Region for all beams on 4th floor
    ops.region(305, '-eleOnly', *beam_tags[4])  # Region for all beams on 5th floor
    ops.region(306, '-eleOnly', *beam_tags[5])  # Region for all beams on 6th floor
    ops.region(307, '-eleOnly', *beam_tags[6])  # Region for all beams on 7th floor
    ops.region(308, '-eleOnly', *beam_tags[7])  # Region for all beams on 8th floor
    ops.region(309, '-eleOnly', *beam_tags[8])  # Region for all beams on 9th floor
    ops.region(310, '-eleOnly', *beam_tags[9])  # Region for all beams on 10th floor
    ops.region(311, '-eleOnly', *beam_tags[10]) # Region for all beams on 11th floor


# Create pvd recorder
record_direc = './pvd/'
os.makedirs(record_direc, exist_ok=True)
ops.recorder('PVD', record_direc, '-precision', 3, '-dT', 1, *['mass', 'reaction'], 'eigen', 10)

# Generate model
build_model()


# ============================================================================
# Eigen Analysis
# ============================================================================
eigen = 1

if eigen:
    ops.wipeAnalysis()
    num_modes = 10

    lambda_list = ops.eigen(num_modes)
    omega_list = [np.sqrt(lam) for lam in lambda_list]
    nat_freq = [np.sqrt(lam)/(2*np.pi) for lam in lambda_list]
    periods = [1/freq for freq in nat_freq]

    print('')
    for ii in range(1, num_modes+1):
        print('Mode {} Tn: {:.2f} sec'.format(ii, periods[ii-1]))

    # Extract translational eigen vector values at COM of each floor
    com_eigen_vec_x = np.zeros(len(com_node_tags))
    com_eigen_vec_y = np.zeros(len(com_node_tags))

    ii = 0
    for key in com_node_tags.keys():
        com_eigen_vec_x[ii] = ops.nodeEigenvector(com_node_tags[key], 1, 1)
        com_eigen_vec_y[ii] = ops.nodeEigenvector(com_node_tags[key], 1, 2)
        ii += 1

    # Normalize eigen vector by the magnitude at the roof
    com_eigen_vec_x /=  com_eigen_vec_x[-1]
    com_eigen_vec_y /=  com_eigen_vec_y[-1]

    modal_prop = ops.modalProperties('-file', 'ModalReport_SMF.txt', '-unorm', '-return')

    # Apply Damping
    damping_ratio = 0.025  # 2.5% Damping

    # Mass and stiffness proportional damping will be applied
    mass_prop_switch = 1.0
    stiff_curr_switch = 1.0
    stiff_comm_switch = 0.0  # Last committed stiffness switch
    stiff_init_switch = 0.0  # Initial stiffness switch

    # Damping coeffieicent will be compusted using the 1st & 5th modes
    omega_i = omega_list[0]  # Angular frequency of 1st Mode
    omega_j = omega_list[4]  # Angular frequency of 5th Mode

    alpha_m = mass_prop_switch * damping_ratio * ((2*omega_i*omega_j) / (omega_i + omega_j))
    beta_k = stiff_curr_switch * damping_ratio * (2 / (omega_i + omega_j))
    beta_k_init = stiff_init_switch * damping_ratio * (2 / (omega_i + omega_j))
    beta_k_comm = stiff_comm_switch * damping_ratio * (2 / (omega_i + omega_j))

    ops.rayleigh(alpha_m, beta_k, beta_k_init, beta_k_comm)


# ============================================================================
# Gravity analysis
# ============================================================================
# Create recorder
grav_direc = './gravity_results/'
os.makedirs(grav_direc, exist_ok=True)

ops.recorder('Node', '-file', grav_direc + 'nodeRxn.txt', '-node', *smf_node_tags['00'].tolist(), '-dof', 3, 'reaction')
ops.recorder('Element', '-file', grav_direc + 'col10_forces.txt', '-ele', 20110, 'force')  # Column 10

num_step_sWgt = 1     # Set weight increments

ops.constraints('Penalty', 1.0e17, 1.0e17)
ops.test('NormDispIncr', 1e-6, 100, 0)
ops.algorithm('KrylovNewton')
ops.numberer('RCM')
ops.system('ProfileSPD')
ops.integrator('LoadControl', 1, 1, 1, 1)
ops.analysis('Static')

ops.analyze(num_step_sWgt)

# Shut down gravity recorders
ops.remove('recorders')


# ============================================================================
# Modal Response Spectrum Analysis
# ============================================================================
mrsa = 1
if mrsa:

    # Load spectral accelerations and periods for response spectrum
    spect_acc = np.loadtxt('../nz_spectral_acc.txt')
    spect_periods = np.loadtxt('../nz_periods.txt')

    direcs = [1, 2]  # Directions for MRSA
    axis = ['X', 'Y']

    # Maintain constant gravity loads and reset time to zero
    ops.loadConst('-time', 0.0)

    for ii in range (len(direcs)):

        # Create directory to save results
        mrsa_res_folder = './mrsa_results/dir' + axis[ii] + '/'
        os.makedirs(mrsa_res_folder, exist_ok=True)

        # Create recorders for beam-response in direction of excitation
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor01_beamResp.txt', '-precision', 16, '-region', 301, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor02_beamResp.txt', '-precision', 16, '-region', 302, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor03_beamResp.txt', '-precision', 16, '-region', 303, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor04_beamResp.txt', '-precision', 16, '-region', 304, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor05_beamResp.txt', '-precision', 16, '-region', 305, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor06_beamResp.txt', '-precision', 16, '-region', 306, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor07_beamResp.txt', '-precision', 16, '-region', 307, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor08_beamResp.txt', '-precision', 16, '-region', 308, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor09_beamResp.txt', '-precision', 16, '-region', 309, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor10_beamResp.txt', '-precision', 16, '-region', 310, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor11_beamResp.txt', '-precision', 16, '-region', 311, 'force')

        # Create recorders to store nodal displacements at the building edges
        ops.recorder('Node', '-file', mrsa_res_folder + 'lowerLeftCornerDisp.txt', '-node', *list(smf_node_tags.loc['col1'])[1:], '-dof', direcs[ii], 'disp')
        ops.recorder('Node', '-file', mrsa_res_folder + 'middleLeftCornerDisp.txt', '-node', *list(smf_node_tags.loc['col13'])[1:], '-dof', direcs[ii], 'disp')
        ops.recorder('Node', '-file', mrsa_res_folder + 'middleCenterCornerDisp.txt', '-node', *list(smf_node_tags.loc['col15'])[1:], '-dof', direcs[ii], 'disp')
        ops.recorder('Node', '-file', mrsa_res_folder + 'upperCenterCornerDisp.txt', '-node', *list(smf_node_tags.loc['col21'])[1:], '-dof', direcs[ii], 'disp')
        ops.recorder('Node', '-file', mrsa_res_folder + 'upperRightCornerDisp.txt', '-node', *list(smf_node_tags.loc['col23'])[1:], '-dof', direcs[ii], 'disp')
        ops.recorder('Node', '-file', mrsa_res_folder + 'lowerRightCornerDisp.txt', '-node', *list(smf_node_tags.loc['col5'])[1:], '-dof', direcs[ii], 'disp')

        # Base shear
        ops.recorder('Node', '-file', mrsa_res_folder + 'baseShear' + axis[ii] + '.txt', '-node', *smf_node_tags['00'].tolist(), '-dof', direcs[ii], 'reaction')

        # Recorders for COM displacement
        ops.recorder('Node', '-file', mrsa_res_folder + 'COM_disp' + axis[ii] + '.txt', '-node', *list(com_node_tags.values()), '-dof', direcs[ii], 'disp')

        for jj in range(num_modes):
            ops.responseSpectrumAnalysis(direcs[ii], '-Tn', *spect_periods, '-Sa', *spect_acc, '-mode', jj + 1)

        # Shut down recorder for current direction of excitation
        ops.remove('recorders')

# Clear model
ops.wipe()

print('\nMRSA completed.')
print('======================================================')


# ============================================================================
# Compute Torsional Irregularity Ratio (TIR)
# ============================================================================
# Obtain peak total response for corner node displacments
# ===== MRSA - X
lower_left_corner_disp = modal_combo(np.loadtxt('./mrsa_results/dirX/lowerLeftCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
mid_left_corner_disp = modal_combo(np.loadtxt('./mrsa_results/dirX/middleLeftCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
mid_center_corner_disp = modal_combo(np.loadtxt('./mrsa_results/dirX/middleCenterCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
upper_center_corner_disp = modal_combo(np.loadtxt('./mrsa_results/dirX/upperCenterCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
upper_right_corner_disp = modal_combo(np.loadtxt('./mrsa_results/dirX/upperRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
lower_right_corner_disp = modal_combo(np.loadtxt('./mrsa_results/dirX/lowerRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)

ax1 = np.maximum(lower_left_corner_disp, mid_left_corner_disp) / (0.5*(lower_left_corner_disp + mid_left_corner_disp))
ax2 = np.maximum(mid_center_corner_disp, upper_center_corner_disp) / (0.5*(mid_center_corner_disp + upper_center_corner_disp))
ax3 = np.maximum(upper_right_corner_disp, lower_right_corner_disp) / (0.5*(upper_right_corner_disp + lower_right_corner_disp))
# b = np.maximum(mid_left_corner_disp, mid_center_corner_disp) / (0.5*(mid_left_corner_disp + mid_center_corner_disp))
# d = np.maximum(upper_center_corner_disp, upper_right_corner_disp) / (0.5*(upper_center_corner_disp + upper_right_corner_disp))
# f = np.maximum(lower_left_corner_disp, lower_right_corner_disp) / (0.5*(lower_left_corner_disp + lower_right_corner_disp))

# ===== MRSA - Y
lower_left_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/lowerLeftCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
mid_left_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/middleLeftCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
mid_center_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/middleCenterCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
upper_center_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/upperCenterCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
upper_right_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/upperRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
lower_right_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/lowerRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)


ay1 = np.maximum(mid_left_corner_dispY, mid_center_corner_dispY) / (0.5*(mid_left_corner_dispY + mid_center_corner_dispY))
ay2 = np.maximum(upper_center_corner_dispY, upper_right_corner_dispY) / (0.5*(upper_center_corner_dispY + upper_right_corner_dispY))
ay3 = np.maximum(lower_left_corner_dispY, lower_right_corner_dispY) / (0.5*(lower_left_corner_dispY + lower_right_corner_dispY))
# a_y = np.maximum(lower_left_corner_dispY, mid_left_corner_dispY) / (0.5*(lower_left_corner_dispY + mid_left_corner_dispY))
# c_y = np.maximum(mid_center_corner_dispY, upper_center_corner_dispY) / (0.5*(mid_center_corner_dispY + upper_center_corner_dispY))
# e_y = np.maximum(upper_right_corner_dispY, lower_right_corner_dispY) / (0.5*(upper_right_corner_dispY + lower_right_corner_dispY))


# # ============================================================================
# # Post-process MRSA results
# # ============================================================================
beam_demands_mrsa_X = process_beam_resp('./mrsa_results/dirX/', lambda_list, damping_ratio, num_modes)
beam_demands_mrsa_Y = process_beam_resp('./mrsa_results/dirY/', lambda_list, damping_ratio, num_modes)

mrsa_base_shearX = modal_combo(np.loadtxt('./mrsa_results/dirX/baseShearX.txt'), lambda_list, damping_ratio, num_modes).sum()
mrsa_base_shearY = modal_combo(np.loadtxt('./mrsa_results/dirY/baseShearY.txt'), lambda_list, damping_ratio, num_modes).sum()

# ============================================================================
# Perform ELF
# ============================================================================
spectral_shape_factor = 0.595
hazard_factor = 0.13
return_per_factor_sls = 0.25
return_per_factor_uls = 1.3
fault_factor = 1.0
perform_factor = 0.7
ductility_factor = 4.0  # SMF
story_weights = np.array(list(total_floor_mass.values())) * grav_metric
seismic_weight = story_weights.sum()

elf_base_shear = nz_horiz_seismic_shear(spectral_shape_factor, hazard_factor,
                                        return_per_factor_sls, return_per_factor_uls,
                                        fault_factor, perform_factor, ductility_factor,
                                        seismic_weight)

elf_force_distrib = nz_horiz_force_distribution(elf_base_shear, story_weights,
                                                np.cumsum(story_heights))


# Deflection amplification factors
kp  = 0.015 + 0.0075*(ductility_factor - 1)
kp = min(max(0.0015, kp), 0.03)

pdelta_fac = (kp * seismic_weight + elf_base_shear) / elf_base_shear  # NZS 1170.5-2004: Sec 7.2.1.2 & 6.5.4.1

drift_modif_fac = 1.5  # NZS 1170.5-2004: Table 7.1

# Compute story drifts
# For MRSA in x-direction
com_dispX = np.loadtxt('./mrsa_results/dirX/COM_dispX.txt')
story_driftX = compute_story_drifts(com_dispX, story_heights, lambda_list, damping_ratio, num_modes)

# For MRSA in y-direction
com_dispY = np.loadtxt('./mrsa_results/dirY/COM_dispY.txt')
story_driftY = compute_story_drifts(com_dispY, story_heights, lambda_list, damping_ratio, num_modes)

# Amplify drifts by required factors
story_driftX *=  (ductility_factor * pdelta_fac * drift_modif_fac)
story_driftY *=  (ductility_factor * pdelta_fac * drift_modif_fac)

max_story_drift = max(story_driftX.max(), story_driftY.max())

# CHECK DRIFT REQUIREMENTS
drift_ok = max_story_drift < 2.5  # MAximum story drift limit = 2.5%

# CHECK STABILITY REQUIREMENTS (P-DELTA)

# CHECK STRENGTH REQUIREMENTS

'========================================================================='
'NEED TO SATISFY DRIFT, STABILITY & STRENGTH REQUIREMENTS BEFORE DOING THIS'
'========================================================================='

# ============================================================================
# Perform static analysis for accidental torsional moment
# ============================================================================
floor_dimen_x = 29.410 * m
floor_dimen_y = 31.025 * m

accid_ecc_x = 0.1 * floor_dimen_x
accid_ecc_y = 0.1 * floor_dimen_y

torsional_mom_x = elf_force_distrib * accid_ecc_y
torsional_mom_y = elf_force_distrib * accid_ecc_x

# AMPLIFY TORSIONAL MOMENT IF REQUIRED BY CODE
# New Zealand does not require amplification of accidental torsional moment

torsional_direc = ['X', 'Y']
torsional_sign = [1, -1]
torsional_folder = ['positive', 'negative']


# Perform static analysis for loading in X & Y direction
for ii in range(len(torsional_direc)):

    # For each direction, account for positive & negative loading
    for jj in range(len(torsional_sign)):
        print('\nNow commencing static analysis using torsional moments for ' + torsional_folder[jj] + ' ' + torsional_direc[ii] + ' direction.')
        build_model()

        print('\nModel generated...')

        # Impose torsional moments at COMs
        com_nodes = list(com_node_tags.values())

        # Assign torsional moments
        ts_tag = 20000
        pattern_tag = 20000

        ops.timeSeries('Constant', ts_tag)
        ops.pattern('Plain', pattern_tag, ts_tag)

        # Loop through each COM node and apply torsional moment
        for kk in range(len(com_nodes)):
            if ii == 1:  # Torsional moment applied about x-axis
                ops.load(com_nodes[kk], 0., 0., 0., torsional_mom_x[kk] * torsional_sign[jj], 0., 0.)
                # print('Moment ' + str(torsional_direc[ii]) + ': ' + str(torsional_mom_x[kk] * torsional_sign[jj]))

            else:  # Torsional moment applied about y-axis
                ops.load(com_nodes[kk], 0., 0., 0., 0., torsional_mom_y[kk] * torsional_sign[jj], 0.)
                # print('Moment ' + str(torsional_direc[ii]) + ': ' + str(torsional_mom_y[kk] * torsional_sign[jj]))


        # Create directory to save results
        accident_torsion_res_folder = './accidental_torsion_results/' + torsional_folder[jj] + torsional_direc[ii] + '/'
        os.makedirs(accident_torsion_res_folder, exist_ok=True)

        # Create recorder for beam-response in direction of static loading
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor01_beamResp.txt', '-precision', 16, '-region', 301, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor02_beamResp.txt', '-precision', 16, '-region', 302, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor03_beamResp.txt', '-precision', 16, '-region', 303, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor04_beamResp.txt', '-precision', 16, '-region', 304, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor05_beamResp.txt', '-precision', 16, '-region', 305, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor06_beamResp.txt', '-precision', 16, '-region', 306, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor07_beamResp.txt', '-precision', 16, '-region', 307, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor08_beamResp.txt', '-precision', 16, '-region', 308, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor09_beamResp.txt', '-precision', 16, '-region', 309, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor10_beamResp.txt', '-precision', 16, '-region', 310, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor11_beamResp.txt', '-precision', 16, '-region', 311, 'force')

        # Base shear
        ops.recorder('Node', '-file', accident_torsion_res_folder + 'baseShearX.txt', '-node',
                      *smf_node_tags['00'].tolist(), '-dof', 1, 2, 4, 5, 'reaction')  # Fx, Fy, Mx, My

        # Perform static analysis
        num_step_sWgt = 1     # Set weight increments

        ops.constraints('Penalty', 1.0e17, 1.0e17)
        ops.test('NormDispIncr', 1e-6, 100, 0)
        ops.algorithm('KrylovNewton')
        ops.numberer('RCM')
        ops.system('ProfileSPD')
        ops.integrator('LoadControl', 1, 1, 1, 1)
        ops.analysis('Static')

        ops.analyze(num_step_sWgt)

        # Shut down recorders
        ops.remove('recorders')

        # Clear model
        ops.wipe()

        print('=============================================================')

print('\nStatic analysis for accidental torsion completed...')
"""


