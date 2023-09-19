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
from helper_functions.get_beam_col_demands import process_beam_col_resp
from helper_functions.get_story_drift import compute_story_drifts
from helper_functions.cqc_modal_combo import modal_combo
from helper_functions.elf_new_zealand import nz_horiz_seismic_shear, nz_horiz_force_distribution
from helper_functions.get_spectral_shape_factor import spectral_shape_fac


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
# Load in New Zealand steel section database
# ============================================================================
nzs_beams = pd.read_excel('../../../nzs_steel_database.xlsx', sheet_name='Beams',
                               index_col='Designation')

nzs_cols = pd.read_excel('../../../nzs_steel_database.xlsx', sheet_name='Columns',
                               index_col='Designation')

nzs_cols = pd.concat([nzs_cols, nzs_beams])
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
pzone_transf_tag_col = 100
pzone_transf_tag_bm_x = 200
pzone_transf_tag_bm_y = 300

# ============================================================================
# Define beam properties
# ============================================================================
bm_nu = 0.28  # Poisson's ratio for steel
bm_E = steel_E
bm_G = bm_E / (2*(1 + bm_nu))

bm_transf_tag_x = 3  # Beams oriented in Global-X direction
bm_transf_tag_y = 4  # Beams oriented in Global-Y direction

# Initialize array of possible values for beam Ix
bm_mom_inertia_strong = np.array(list(nzs_beams['Ix']))

# The geometric properties of the beams will be defined relative to the stiffness of the first floor beam
base_Ix = 9975.573722316492  # No need to multiply by 'mm' or '1E6'
slope_Ix_line = 0.004586197032942164
col_group_heights = np.array([0, 6.2, 15.5, 24.8, 31])  # Height of column groups from the 1st floor

# Assume linear relationship
# Base Ix & slope
# bm_Ix_modif = [1, 1, 1, 1, 1]
# bm_Ix_modif = [1, 0.8, 0.6, 0.5, 0.4]
bm_Ix_modif = 1 - slope_Ix_line*col_group_heights

bm_sect_flr_1 = nzs_beams.loc[nzs_beams.index[nzs_beams['Ix'] >= bm_Ix_modif[0] * base_Ix].tolist()[-1]]
bm_sect_flr_2_to_4 = nzs_beams.loc[nzs_beams.index[nzs_beams['Ix'] >= bm_Ix_modif[1] * base_Ix].tolist()[-1]]
bm_sect_flr_5_to_7 = nzs_beams.loc[nzs_beams.index[nzs_beams['Ix'] >= bm_Ix_modif[2] * base_Ix].tolist()[-1]]
bm_sect_flr_8_to_10 = nzs_beams.loc[nzs_beams.index[nzs_beams['Ix'] >= bm_Ix_modif[3] * base_Ix].tolist()[-1]]
bm_sect_flr_11 = nzs_beams.loc[nzs_beams.index[nzs_beams['Ix'] >= bm_Ix_modif[4] * base_Ix].tolist()[-1]]

bm_sections = [bm_sect_flr_1.name, bm_sect_flr_2_to_4.name, bm_sect_flr_5_to_7.name, bm_sect_flr_8_to_10.name, bm_sect_flr_11.name]

bm_prop_flr_1 = [bm_sect_flr_1, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_2_to_4 = [bm_sect_flr_2_to_4, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_5_to_7 = [bm_sect_flr_5_to_7, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_8_to_10 = [bm_sect_flr_8_to_10, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
bm_prop_flr_11 = [bm_sect_flr_11, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]

# ============================================================================
# Define column properties
# ============================================================================
col_nu = 0.28  # Poisson's ratio for steel
col_E = steel_E
col_G = col_E / (2*(1 + col_nu))

col_transf_tag_EW = 1
col_transf_tag_NS = 2

col_beam_mom_ratio = 1.25

col_sect_flr_1 = nzs_cols.loc[nzs_cols.index[nzs_cols['Zx'] >= col_beam_mom_ratio * bm_sect_flr_1['Zx']].tolist()[-1]]

# Keep the same column designation up the building height, while changing to lighter sections
col_designation = col_sect_flr_1.name

desig = ''
for char in col_designation:
    desig += char
    if char == 'B' or char == 'C':
        break

col_sect_flr_2_to_4 = nzs_cols.loc[nzs_cols.index[nzs_cols['Zx'] >= col_beam_mom_ratio * bm_sect_flr_2_to_4['Zx']].tolist()[-1]]
col_sect_flr_5_to_7 = nzs_cols.loc[nzs_cols.index[nzs_cols['Zx'] >= col_beam_mom_ratio * bm_sect_flr_5_to_7['Zx']].tolist()[-1]]
col_sect_flr_8_to_10 = nzs_cols.loc[nzs_cols.index[nzs_cols['Zx'] >= col_beam_mom_ratio * bm_sect_flr_8_to_10['Zx']].tolist()[-1]]
col_sect_flr_11 = nzs_cols.loc[nzs_cols.index[nzs_cols['Zx'] >= col_beam_mom_ratio * bm_sect_flr_11['Zx']].tolist()[-1]]

col_sections = [col_sect_flr_1.name, col_sect_flr_2_to_4.name, col_sect_flr_5_to_7.name, col_sect_flr_8_to_10.name, col_sect_flr_11.name]


col_prop_flr_1 = [col_sect_flr_1, col_E, col_G, col_transf_tag_EW, col_transf_tag_NS, pzone_transf_tag_col]
col_prop_flr_2_to_4 = [col_sect_flr_2_to_4, col_E, col_G, col_transf_tag_EW, col_transf_tag_NS, pzone_transf_tag_col]
col_prop_5_to_7 = [col_sect_flr_5_to_7, col_E, col_G, col_transf_tag_EW, col_transf_tag_NS, pzone_transf_tag_col]
col_prop_8_to_10 = [col_sect_flr_8_to_10, col_E, col_G, col_transf_tag_EW, col_transf_tag_NS, pzone_transf_tag_col]
col_prop_flr_11 = [col_sect_flr_11, col_E, col_G, col_transf_tag_EW, col_transf_tag_NS, pzone_transf_tag_col]


# ============================================================================
# Initialize dictionary to store node tags of COM for all floors
# Initialize dictionary to store total mass of each floor
# ============================================================================
com_node_tags = {}
total_floor_mass = {}

# ============================================================================
# Define function to create a floor
# ============================================================================

def create_floor(elev, floor_num, beam_prop=None, col_prop=None, floor_label='',):

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
        create_columns(floor_num, smf_node_tags, col_prop, beam_prop)
        create_beams(floor_num, elev, com_node, smf_node_tags, smf_coords_df, beam_prop, col_prop[0])

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
    create_floor(flr1, '01', bm_prop_flr_1, col_prop_flr_1, '1st')
    create_floor(flr2, '02', bm_prop_flr_2_to_4, col_prop_flr_2_to_4, '2nd')
    create_floor(flr3, '03', bm_prop_flr_2_to_4, col_prop_flr_2_to_4, '3rd')
    create_floor(flr4, '04', bm_prop_flr_2_to_4, col_prop_flr_2_to_4, '4th')
    create_floor(flr5, '05', bm_prop_flr_5_to_7, col_prop_5_to_7, '5th')
    create_floor(flr6, '06', bm_prop_flr_5_to_7, col_prop_5_to_7, '6th')
    create_floor(flr7, '07', bm_prop_flr_5_to_7, col_prop_5_to_7, '7th')
    create_floor(flr8, '08', bm_prop_flr_8_to_10, col_prop_8_to_10, '8th')
    create_floor(flr9, '09', bm_prop_flr_8_to_10, col_prop_8_to_10, '9th')
    create_floor(flr10, '10', bm_prop_flr_8_to_10, col_prop_8_to_10, '10th')
    create_floor(roof_flr, '11', bm_prop_flr_11, col_prop_flr_11, 'Roof')

    # ============================================================================
    # Create regions for SMF beams & columns based on floor
    # ============================================================================
    # Get all element tags
    elem_tags = ops.getEleTags()

    floor_nums = ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11']

    beam_tags = []
    col_tags = []

    for floor in floor_nums:
        floor_bm_tags = []
        floor_col_tags = []

        for tag in elem_tags:

            # Only select beam elements
            if str(tag).startswith('2' + floor):
                floor_bm_tags.append(tag)

            # Only select column elements
            if str(tag).startswith('3' + floor):
                floor_col_tags.append(tag)

        beam_tags.append(floor_bm_tags)
        col_tags.append(floor_col_tags)

    # Beams
    ops.region(201, '-eleOnly', *beam_tags[0])  # Region for all beams on 1st floor
    ops.region(202, '-eleOnly', *beam_tags[1])  # Region for all beams on 2nd floor
    ops.region(203, '-eleOnly', *beam_tags[2])  # Region for all beams on 3rd floor
    ops.region(204, '-eleOnly', *beam_tags[3])  # Region for all beams on 4th floor
    ops.region(205, '-eleOnly', *beam_tags[4])  # Region for all beams on 5th floor
    ops.region(206, '-eleOnly', *beam_tags[5])  # Region for all beams on 6th floor
    ops.region(207, '-eleOnly', *beam_tags[6])  # Region for all beams on 7th floor
    ops.region(208, '-eleOnly', *beam_tags[7])  # Region for all beams on 8th floor
    ops.region(209, '-eleOnly', *beam_tags[8])  # Region for all beams on 9th floor
    ops.region(210, '-eleOnly', *beam_tags[9])  # Region for all beams on 10th floor
    ops.region(211, '-eleOnly', *beam_tags[10]) # Region for all beams on 11th floor

    # Columns
    ops.region(301, '-eleOnly', *col_tags[0])  # Region for all columns on 1st floor
    ops.region(302, '-eleOnly', *col_tags[1])  # Region for all columns on 2nd floor
    ops.region(303, '-eleOnly', *col_tags[2])  # Region for all columns on 3rd floor
    ops.region(304, '-eleOnly', *col_tags[3])  # Region for all columns on 4th floor
    ops.region(305, '-eleOnly', *col_tags[4])  # Region for all columns on 5th floor
    ops.region(306, '-eleOnly', *col_tags[5])  # Region for all columns on 6th floor
    ops.region(307, '-eleOnly', *col_tags[6])  # Region for all columns on 7th floor
    ops.region(308, '-eleOnly', *col_tags[7])  # Region for all columns on 8th floor
    ops.region(309, '-eleOnly', *col_tags[8])  # Region for all columns on 9th floor
    ops.region(310, '-eleOnly', *col_tags[9])  # Region for all columns on 10th floor
    ops.region(311, '-eleOnly', *col_tags[10]) # Region for all columns on 11th floor

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

ops.recorder('Node', '-file', grav_direc + 'nodeRxn.txt', '-node', *smf_node_tags['00'].tolist(), '-dof', 1, 2, 3, 4, 5, 6, 'reaction')
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
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor01_beamResp.txt', '-precision', 9, '-region', 201, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor02_beamResp.txt', '-precision', 9, '-region', 202, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor03_beamResp.txt', '-precision', 9, '-region', 203, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor04_beamResp.txt', '-precision', 9, '-region', 204, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor05_beamResp.txt', '-precision', 9, '-region', 205, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor06_beamResp.txt', '-precision', 9, '-region', 206, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor07_beamResp.txt', '-precision', 9, '-region', 207, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor08_beamResp.txt', '-precision', 9, '-region', 208, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor09_beamResp.txt', '-precision', 9, '-region', 209, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor10_beamResp.txt', '-precision', 9, '-region', 210, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor11_beamResp.txt', '-precision', 9, '-region', 211, 'force')

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


        # Create recorders to store nodal displacements at the building edges
        ops.recorder('Node', '-file', mrsa_res_folder + 'lowerLeftCornerDisp.txt', '-node', *list(smf_node_tags.loc['col1'])[1:], '-dof', direcs[ii], 'disp')
        ops.recorder('Node', '-file', mrsa_res_folder + 'middleLeftCornerDisp.txt', '-node', *list(smf_node_tags.loc['col13'])[1:], '-dof', direcs[ii], 'disp')
        ops.recorder('Node', '-file', mrsa_res_folder + 'middleCenterCornerDisp.txt', '-node', *list(smf_node_tags.loc['col15'])[1:], '-dof', direcs[ii], 'disp')
        ops.recorder('Node', '-file', mrsa_res_folder + 'upperLeftCornerDisp.txt', '-node', *list(smf_node_tags.loc['col21'])[1:], '-dof', direcs[ii], 'disp')
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
upper_left_corner_disp = modal_combo(np.loadtxt('./mrsa_results/dirX/upperLeftCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
upper_right_corner_disp = modal_combo(np.loadtxt('./mrsa_results/dirX/upperRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
lower_right_corner_disp = modal_combo(np.loadtxt('./mrsa_results/dirX/lowerRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)

tir_x_edgeA = np.maximum(lower_left_corner_disp, mid_left_corner_disp) / (0.5*(lower_left_corner_disp + mid_left_corner_disp))
tir_x_edgeB = np.maximum(mid_left_corner_disp, mid_center_corner_disp) / (0.5*(mid_left_corner_disp + mid_center_corner_disp))
tir_x_edgeC = np.maximum(mid_center_corner_disp, upper_left_corner_disp) / (0.5*(mid_center_corner_disp + upper_left_corner_disp))
tir_x_edgeD = np.maximum(upper_left_corner_disp, upper_right_corner_disp) / (0.5*(upper_left_corner_disp + upper_right_corner_disp))
tir_x_edgeE = np.maximum(upper_right_corner_disp, lower_right_corner_disp) / (0.5*(upper_right_corner_disp + lower_right_corner_disp))
tir_x_edgeF = np.maximum(lower_left_corner_disp, lower_right_corner_disp) / (0.5*(lower_left_corner_disp + lower_right_corner_disp))

# ===== MRSA - Y
lower_left_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/lowerLeftCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
mid_left_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/middleLeftCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
mid_center_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/middleCenterCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
upper_left_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/upperLeftCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
upper_right_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/upperRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
lower_right_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/lowerRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)

tir_y_edgeA = np.maximum(lower_left_corner_dispY, mid_left_corner_dispY) / (0.5*(lower_left_corner_dispY + mid_left_corner_dispY))
tir_y_edgeB = np.maximum(mid_left_corner_dispY, mid_center_corner_dispY) / (0.5*(mid_left_corner_dispY + mid_center_corner_dispY))
tir_y_edgeC = np.maximum(mid_center_corner_dispY, upper_left_corner_dispY) / (0.5*(mid_center_corner_dispY + upper_left_corner_dispY))
tir_y_edgeD = np.maximum(upper_left_corner_dispY, upper_right_corner_dispY) / (0.5*(upper_left_corner_dispY + upper_right_corner_dispY))
tir_y_edgeE = np.maximum(upper_right_corner_dispY, lower_right_corner_dispY) / (0.5*(upper_right_corner_dispY + lower_right_corner_dispY))
tir_y_edgeF = np.maximum(lower_left_corner_dispY, lower_right_corner_dispY) / (0.5*(lower_left_corner_dispY + lower_right_corner_dispY))

# ============================================================================
# Post-process MRSA results
# ============================================================================
mrsa_base_shearX = modal_combo(np.loadtxt('./mrsa_results/dirX/baseShearX.txt'), lambda_list, damping_ratio, num_modes).sum()
mrsa_base_shearY = modal_combo(np.loadtxt('./mrsa_results/dirY/baseShearY.txt'), lambda_list, damping_ratio, num_modes).sum()

# ============================================================================
# Perform ELF
# ============================================================================
spectral_shape_factor = spectral_shape_fac(periods[0])
hazard_factor = 0.13
return_per_factor_sls = 0.25
return_per_factor_uls = 1.3
fault_factor = 1.0
perform_factor = 0.7
ductility_factor = 4.0  # SMF  4.0
story_weights = np.array(list(total_floor_mass.values())) * grav_metric
seismic_weight = story_weights.sum()

elf_base_shear = nz_horiz_seismic_shear(spectral_shape_factor, hazard_factor,
                                        return_per_factor_sls, return_per_factor_uls,
                                        fault_factor, perform_factor, ductility_factor,
                                        seismic_weight)

elf_force_distrib = nz_horiz_force_distribution(elf_base_shear, story_weights,
                                                np.cumsum(story_heights))

# Compute factors for scaling MRSA demands to ELF demands NZS 1170.5:2004 - Sect. 5.2.2.2b
elf_mrsaX_scale_factor = max(elf_base_shear / mrsa_base_shearX, 1.0)
elf_mrsaY_scale_factor = max(elf_base_shear / mrsa_base_shearY, 1.0)

# ============================================================================
# Check drift and stability requirements
# ============================================================================

# Deflection amplification factors
kp  = 0.015 + 0.0075*(ductility_factor - 1)
kp = min(max(0.0015, kp), 0.03)

# pdelta_fac = (kp * seismic_weight + elf_base_shear) / elf_base_shear  # NZS 1170.5-2004: Sec 7.2.1.2 & 6.5.4.1
pdelta_fac = 1
drift_modif_fac = 1.5  # NZS 1170.5-2004: Table 7.1

# Compute story drifts
# For MRSA in x-direction
com_dispX = np.loadtxt('./mrsa_results/dirX/COM_dispX.txt')
story_driftX = compute_story_drifts(com_dispX, story_heights, lambda_list, damping_ratio, num_modes)

# For MRSA in y-direction
com_dispY = np.loadtxt('./mrsa_results/dirY/COM_dispY.txt')
story_driftY = compute_story_drifts(com_dispY, story_heights, lambda_list, damping_ratio, num_modes)

# Amplify drifts by required factors
story_driftX *=  (elf_mrsaX_scale_factor * ductility_factor * pdelta_fac * drift_modif_fac)
story_driftY *=  (elf_mrsaY_scale_factor * ductility_factor * pdelta_fac * drift_modif_fac)

# CHECK DRIFT REQUIREMENTS
max_story_drift = max(story_driftX.max(), story_driftY.max())
drift_ok = max_story_drift < 2.5  # Maximum story drift limit = 2.5%  NZS 1170.5:2004 - Sect 7.5.1

print('\nMaximum story drift: {:.2f}%'.format(max_story_drift))
if drift_ok:
    print('Story drift requirements satisfied.')
else:
    print('Story drift requirements NOT satisfied.')

# CHECK STABILITY REQUIREMENTS (P-DELTA) NZS 1170.5:2004 - Sect 6.5.1
thetaX = story_weights * 0.01 * story_driftX / (elf_force_distrib * story_heights)
thetaY = story_weights * 0.01 * story_driftY / (elf_force_distrib * story_heights)

max_theta = max(thetaX.max(), thetaY.max())
theta_ok = max_theta < 0.3

print('\nMaximum stability coefficient: {:.2f}'.format(max_theta))
if theta_ok:
    print('Stability requirements satisfied.')
else:
    print('Stability requirements NOT satisfied.')


print('\nBeam sections: ', bm_sections)
print('\nColumn sections: ', col_sections)

# CHECK STRENGTH REQUIREMENTS



'NEED TO SATISFY DRIFT, STABILITY & STRENGTH REQUIREMENTS BEFORE DOING THIS'
# ============================================================================
# Perform static analysis for accidental torsional moment
# ============================================================================
floor_dimen_x = 29.410 * m
floor_dimen_y = 31.025 * m

accid_ecc_x = floor_dimen_x / 10
accid_ecc_y = floor_dimen_y / 10

torsional_mom_x = elf_force_distrib * accid_ecc_y
torsional_mom_y = elf_force_distrib * accid_ecc_x

# AMPLIFY TORSIONAL MOMENT IF REQUIRED BY CODE
# New Zealand does not require amplification of accidental torsional moment

torsional_direc = ['X', 'Y']
elf_dof = [1, 2]
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

        # # Loop through each COM node and apply torsional moment
        for kk in range(len(com_nodes)):
            if torsional_direc[ii] == 'X':  # Torsional moment applied about x-axis
                ops.load(com_nodes[kk], 0., 0., 0., 0., 0., torsional_mom_x[kk] * torsional_sign[jj])

            else:  # Torsional moment applied about y-axis
                ops.load(com_nodes[kk], 0., 0., 0., 0., 0., torsional_mom_y[kk] * torsional_sign[jj])

        # Create directory to save results
        accident_torsion_res_folder = './accidental_torsion_results/' + torsional_folder[jj] + torsional_direc[ii] + '/'
        os.makedirs(accident_torsion_res_folder, exist_ok=True)

        # Create recorder for beam-response in direction of static loading
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor01_beamResp.txt', '-precision', 9, '-region', 201, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor02_beamResp.txt', '-precision', 9, '-region', 202, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor03_beamResp.txt', '-precision', 9, '-region', 203, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor04_beamResp.txt', '-precision', 9, '-region', 204, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor05_beamResp.txt', '-precision', 9, '-region', 205, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor06_beamResp.txt', '-precision', 9, '-region', 206, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor07_beamResp.txt', '-precision', 9, '-region', 207, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor08_beamResp.txt', '-precision', 9, '-region', 208, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor09_beamResp.txt', '-precision', 9, '-region', 209, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor10_beamResp.txt', '-precision', 9, '-region', 210, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor11_beamResp.txt', '-precision', 9, '-region', 211, 'force')

        # Create recorders for column response direction of static loading
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor01_colResp.txt', '-precision', 9, '-region', 301, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor02_colResp.txt', '-precision', 9, '-region', 302, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor03_colResp.txt', '-precision', 9, '-region', 303, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor04_colResp.txt', '-precision', 9, '-region', 304, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor05_colResp.txt', '-precision', 9, '-region', 305, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor06_colResp.txt', '-precision', 9, '-region', 306, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor07_colResp.txt', '-precision', 9, '-region', 307, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor08_colResp.txt', '-precision', 9, '-region', 308, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor09_colResp.txt', '-precision', 9, '-region', 309, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor10_colResp.txt', '-precision', 9, '-region', 310, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor11_colResp.txt', '-precision', 9, '-region', 311, 'force')

        # Base shear
        ops.recorder('Node', '-file', accident_torsion_res_folder + 'baseShear' + torsional_direc[ii] + '.txt', '-node',
                      *smf_node_tags['00'].tolist(), '-dof', elf_dof[ii], 'reaction')  # Fx or Fy

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

# ============================================================================
# Post-process MRSA & accidental torsion results
# ============================================================================
beam_demandsX = process_beam_col_resp('beam', './mrsa_results/dirX/', './accidental_torsion_results/positiveX/',
                                      './accidental_torsion_results/negativeX/', lambda_list, damping_ratio,
                                      num_modes, elf_mrsaX_scale_factor, pdelta_fac)

beam_demandsY = process_beam_col_resp('beam', './mrsa_results/dirY/', './accidental_torsion_results/positiveY/',
                                      './accidental_torsion_results/negativeY/', lambda_list, damping_ratio,
                                      num_modes, elf_mrsaY_scale_factor, pdelta_fac)

col_demandsX = process_beam_col_resp('col', './mrsa_results/dirX/', './accidental_torsion_results/positiveX/',
                                     './accidental_torsion_results/negativeX/', lambda_list, damping_ratio,
                                     num_modes, elf_mrsaX_scale_factor, pdelta_fac)

col_demandsY = process_beam_col_resp('col', './mrsa_results/dirY/', './accidental_torsion_results/positiveY/',
                                     './accidental_torsion_results/negativeY/', lambda_list, damping_ratio, num_modes,
                                     elf_mrsaY_scale_factor, pdelta_fac)

# Base shear due to static accidental torsion analysis
# These should basically equal zero, as only moments are applied about the z-axis for the static analyses.
accid_torsion_baseShear_pos_X = np.loadtxt('./accidental_torsion_results/positiveX/baseShearX.txt').sum()
accid_torsion_baseShear_neg_X = np.loadtxt('./accidental_torsion_results/negativeX/baseShearX.txt').sum()

accid_torsion_baseShear_pos_Y = np.loadtxt('./accidental_torsion_results/positiveY/baseShearY.txt').sum()
accid_torsion_baseShear_neg_Y = np.loadtxt('./accidental_torsion_results/negativeY/baseShearY.txt').sum()

base_shearX = max((mrsa_base_shearX + accid_torsion_baseShear_pos_X), (mrsa_base_shearX + accid_torsion_baseShear_neg_X))
base_shearY = max((mrsa_base_shearY + accid_torsion_baseShear_pos_Y), (mrsa_base_shearY + accid_torsion_baseShear_neg_Y))
