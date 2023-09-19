# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:43:33 2023

@author: Uzo Uwaoma
"""
# Import required modules
import os
import sys
import time
import random
import openseespy.opensees as ops
import pandas as pd
import numpy as np

# Append directory of helper functions to Pyhton Path
sys.path.append('../')

from helper_functions.create_floor_shell import refine_mesh
from helper_functions.create_floor_shell import create_shell
from helper_functions.build_smf_column import create_columns
from helper_functions.build_smf_beam import create_beams
from helper_functions.cqc_modal_combo import modal_combo
from helper_functions.get_story_drift import compute_story_drifts
from helper_functions.elf_new_zealand import nz_horiz_seismic_shear
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
# Define beam material properties
# ============================================================================
bm_nu = 0.28  # Poisson's ratio for steel
bm_E = steel_E
bm_G = bm_E / (2*(1 + bm_nu))

bm_transf_tag_x = 3  # Beams oriented in Global-X direction
bm_transf_tag_y = 4  # Beams oriented in Global-Y direction

# ============================================================================
# Define column material properties
# ============================================================================
col_nu = 0.28  # Poisson's ratio for steel
col_E = steel_E
col_G = col_E / (2*(1 + col_nu))

col_transf_tag_EW = 1
col_transf_tag_NS = 2

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

    # print('Floor ' + floor_num + ' created')

# ============================================================================
# Model builder
# ============================================================================


def build_model(optim_params):

    print(optim_params)

    # The geometric properties of the beams will be defined relative to the stiffness of the first floor beam
    base_Ix = optim_params[0]  # No need to multiply by 'mm' or '1E6'
    slope_Ix_line = optim_params[1]

    col_group_heights = np.array([0, 6.2, 15.5, 24.8, 31])  # Height of column groups from the 1st floor

    # Assume linear relationship
    beam_Ix_distrib = 1 - slope_Ix_line*col_group_heights

    # ============================================================================
    # Define beam section properties
    # ============================================================================
    bm_sect_flr_1 = nzs_beams.loc[nzs_beams.index[nzs_beams['Ix'] >= beam_Ix_distrib[0] * base_Ix].tolist()[-1]]
    bm_sect_flr_2_to_4 = nzs_beams.loc[nzs_beams.index[nzs_beams['Ix'] >= beam_Ix_distrib[1] * base_Ix].tolist()[-1]]
    bm_sect_flr_5_to_7 = nzs_beams.loc[nzs_beams.index[nzs_beams['Ix'] >= beam_Ix_distrib[2] * base_Ix].tolist()[-1]]
    bm_sect_flr_8_to_10 = nzs_beams.loc[nzs_beams.index[nzs_beams['Ix'] >= beam_Ix_distrib[3] * base_Ix].tolist()[-1]]
    bm_sect_flr_11 = nzs_beams.loc[nzs_beams.index[nzs_beams['Ix'] >= beam_Ix_distrib[4] * base_Ix].tolist()[-1]]


    bm_sections = [bm_sect_flr_1.name, bm_sect_flr_2_to_4.name, bm_sect_flr_5_to_7.name, bm_sect_flr_8_to_10.name, bm_sect_flr_11.name]
    print('Beam sections: ', bm_sections)

    bm_prop_flr_1 = [bm_sect_flr_1, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
    bm_prop_flr_2_to_4 = [bm_sect_flr_2_to_4, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
    bm_prop_flr_5_to_7 = [bm_sect_flr_5_to_7, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
    bm_prop_flr_8_to_10 = [bm_sect_flr_8_to_10, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
    bm_prop_flr_11 = [bm_sect_flr_11, bm_E, bm_G, bm_transf_tag_x, bm_transf_tag_y, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]


    # ============================================================================
    # Define column section properties
    # ============================================================================
    col_beam_mom_ratio = 1.25  # Overstrength factor per NZS 3404.1:1997 - Table 12.2.8(1)

    col_sect_flr_1 = nzs_cols.loc[nzs_cols.index[nzs_cols['Zx'] >= col_beam_mom_ratio * bm_sect_flr_1['Zx']].tolist()[-1]]
    col_sect_flr_2_to_4 = nzs_cols.loc[nzs_cols.index[nzs_cols['Zx'] >= col_beam_mom_ratio * bm_sect_flr_2_to_4['Zx']].tolist()[-1]]
    col_sect_flr_5_to_7 = nzs_cols.loc[nzs_cols.index[nzs_cols['Zx'] >= col_beam_mom_ratio * bm_sect_flr_5_to_7['Zx']].tolist()[-1]]
    col_sect_flr_8_to_10 = nzs_cols.loc[nzs_cols.index[nzs_cols['Zx'] >= col_beam_mom_ratio * bm_sect_flr_8_to_10['Zx']].tolist()[-1]]
    col_sect_flr_11 = nzs_cols.loc[nzs_cols.index[nzs_cols['Zx'] >= col_beam_mom_ratio * bm_sect_flr_11['Zx']].tolist()[-1]]

    col_sections = [col_sect_flr_1.name, col_sect_flr_2_to_4.name, col_sect_flr_5_to_7.name, col_sect_flr_8_to_10.name, col_sect_flr_11.name]
    print('Column sections: ', col_sections)

    col_prop_flr_1 = [col_sect_flr_1, col_E, col_G, col_transf_tag_EW, col_transf_tag_NS, pzone_transf_tag_col]
    col_prop_flr_2_to_4 = [col_sect_flr_2_to_4, col_E, col_G, col_transf_tag_EW, col_transf_tag_NS, pzone_transf_tag_col]
    col_prop_5_to_7 = [col_sect_flr_5_to_7, col_E, col_G, col_transf_tag_EW, col_transf_tag_NS, pzone_transf_tag_col]
    col_prop_8_to_10 = [col_sect_flr_8_to_10, col_E, col_G, col_transf_tag_EW, col_transf_tag_NS, pzone_transf_tag_col]
    col_prop_flr_11 = [col_sect_flr_11, col_E, col_G, col_transf_tag_EW, col_transf_tag_NS, pzone_transf_tag_col]


    # Clear model
    # ops.wipe()

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
    # print('Now creating SSMF model... \n')

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
            if str(tag).startswith('2' + floor):
                floor_bm_tags.append(tag)

        beam_tags.append(floor_bm_tags)

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

    # ============================================================================
    # Gravity analysis
    # ============================================================================
    ops.wipeAnalysis()
    ops.loadConst('-time', 0.0)
    num_step_sWgt = 1     # Set weight increments

    ops.constraints('Penalty', 1.0e17, 1.0e17)
    ops.test('NormDispIncr', 1e-6, 100, 0)
    ops.algorithm('KrylovNewton')
    ops.numberer('RCM')
    ops.system('ProfileSPD')
    ops.integrator('LoadControl', 1, 1, 1, 1)
    ops.analysis('Static')

    ops.analyze(num_step_sWgt)

    # ============================================================================
    # Eigen Analysis
    # ============================================================================
    ops.wipeAnalysis()
    ops.loadConst('-time', 0.0)

    num_modes = 10

    lambda_list = ops.eigen(num_modes)
    omega_list = [np.sqrt(lam) for lam in lambda_list]
    nat_freq = [np.sqrt(lam)/(2*np.pi) for lam in lambda_list]
    periods = [1/freq for freq in nat_freq]

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

    ops.modalProperties()

    # ============================================================================
    # Modal Response Spectrum Analysis
    # ============================================================================

    # Load spectral accelerations and periods for response spectrum
    spect_acc = np.loadtxt('../nz_spectral_acc.txt')
    spect_periods = np.loadtxt('../nz_periods.txt')

    direcs = [1, 2]  # Directions for MRSA
    axis = ['X', 'Y']

    # Maintain constant gravity loads and reset time to zero
    ops.loadConst('-time', 0.0)

    for ii in range (len(direcs)):

        # Create directory to save results
        mrsa_res_folder = './optimization_results/mrsa_results/dir' + axis[ii] + '/'
        os.makedirs(mrsa_res_folder, exist_ok=True)

        # Create recorders for beam-response in direction of excitation
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor01_beamResp.txt', '-precision', 16, '-region', 201, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor02_beamResp.txt', '-precision', 16, '-region', 202, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor03_beamResp.txt', '-precision', 16, '-region', 203, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor04_beamResp.txt', '-precision', 16, '-region', 204, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor05_beamResp.txt', '-precision', 16, '-region', 205, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor06_beamResp.txt', '-precision', 16, '-region', 206, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor07_beamResp.txt', '-precision', 16, '-region', 207, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor08_beamResp.txt', '-precision', 16, '-region', 208, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor09_beamResp.txt', '-precision', 16, '-region', 209, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor10_beamResp.txt', '-precision', 16, '-region', 210, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor11_beamResp.txt', '-precision', 16, '-region', 211, 'force')

        # Base shear
        ops.recorder('Node', '-file', mrsa_res_folder + 'baseShear' + axis[ii] + '.txt', '-node', *smf_node_tags['00'].tolist(), '-dof', direcs[ii], 'reaction')

        # Recorders for COM displacement
        ops.recorder('Node', '-file', mrsa_res_folder + 'COM_disp' + axis[ii] + '.txt', '-node', *list(com_node_tags.values()), '-dof', direcs[ii], 'disp')

        for jj in range(num_modes):
            ops.responseSpectrumAnalysis(direcs[ii], '-Tn', *spect_periods, '-Sa', *spect_acc, '-mode', jj + 1)

        # Shut down recorder for current direction of excitation
        ops.remove('recorders')

    # # Clear model
    ops.wipe()

    # ============================================================================
    # Post-process MRSA results
    # ============================================================================
    mrsa_base_shearX = modal_combo(np.loadtxt('./optimization_results/mrsa_results/dirX/baseShearX.txt'), lambda_list, damping_ratio, num_modes).sum()
    mrsa_base_shearY = modal_combo(np.loadtxt('./optimization_results/mrsa_results/dirY/baseShearY.txt'), lambda_list, damping_ratio, num_modes).sum()

    # ============================================================================
    # Perform ELF
    # ============================================================================
    spectral_shape_factor = spectral_shape_fac(periods[0])
    hazard_factor = 0.13
    return_per_factor_sls = 0.25
    return_per_factor_uls = 1.3
    fault_factor = 1.0
    perform_factor = 0.7
    ductility_factor = 4.0  # Category 1 structure
    story_weights = np.array(list(total_floor_mass.values())) * grav_metric
    seismic_weight = story_weights.sum()

    elf_base_shear = nz_horiz_seismic_shear(spectral_shape_factor, hazard_factor,
                                            return_per_factor_sls, return_per_factor_uls,
                                            fault_factor, perform_factor, ductility_factor,
                                            seismic_weight)

    # Compute factors for scaling MRSA demands to ELF demands NZS 1170.5:2004 - Sect. 5.2.2.2b
    elf_mrsaX_scale_factor = max(elf_base_shear / mrsa_base_shearX, 1.0)
    elf_mrsaY_scale_factor = max(elf_base_shear / mrsa_base_shearY, 1.0)

    # ============================================================================
    # Check drift and stability requirements
    # ============================================================================

    # Deflection amplification factors
    kp  = 0.015 + 0.0075*(ductility_factor - 1)
    kp = min(max(0.0015, kp), 0.03)

    pdelta_fac = (kp * seismic_weight + elf_base_shear) / elf_base_shear  # NZS 1170.5-2004: Sec 7.2.1.2 & 6.5.4.1
    # pdelta_fac = 1

    drift_modif_fac = 1.5  # NZS 1170.5-2004: Table 7.1

    # Compute story drifts
    # For MRSA in x-direction
    com_dispX = np.loadtxt('./optimization_results/mrsa_results/dirX/COM_dispX.txt')
    story_driftX = compute_story_drifts(com_dispX, story_heights, lambda_list, damping_ratio, num_modes)

    # For MRSA in y-direction
    com_dispY = np.loadtxt('./optimization_results/mrsa_results/dirY/COM_dispY.txt')
    story_driftY = compute_story_drifts(com_dispY, story_heights, lambda_list, damping_ratio, num_modes)

    # Amplify drifts by required factors
    story_driftX *=  (elf_mrsaX_scale_factor * ductility_factor * pdelta_fac * drift_modif_fac)
    story_driftY *=  (elf_mrsaY_scale_factor * ductility_factor * pdelta_fac * drift_modif_fac)

    # CHECK DRIFT REQUIREMENTS
    max_story_drift = max(story_driftX.max(), story_driftY.max())

    return max_story_drift


# Initialize array of possible values for beam Ix
bm_Ix_vals = np.array(list(nzs_beams['Ix']))
bm_Zx_vals = np.array(list(nzs_beams['Zx']))

# Maximum beam section will be a function of Zx to enable capacity design
max_bm_Zx = bm_Zx_vals.max() / 1.25

# Define bounds on possible values for first floor beam Ix.
bm_Ix_min = bm_Ix_vals.min()
bm_Ix_max = bm_Ix_vals[bm_Zx_vals <= max_bm_Zx][0]
bm_Ix_bounds = (bm_Ix_min, bm_Ix_max)

# Define bounds on slope of Ix distribution
bm_Ix_slope_min = 1 / 310
bm_Ix_slope_max = 1 / 31
bm_Ix_slope_bounds = (bm_Ix_slope_min, bm_Ix_slope_max)


'*********************************************************************************************'
'*********************************************************************************************'
'*********************************************************************************************'

init_time = time.time()

nv = 2  # number of variables
optim = -1  # if minimization problem, optim = -1; if maximization problem, optim = 1

bounds = [bm_Ix_bounds, bm_Ix_slope_bounds]

particle_size = 50 * nv  # number of particles
iterations = 50  # max number of iterations

w = 0.9   # inertia constant  0.75
c1 = 0.5  # cognitive constant
c2 = 2    # social constant

# END OF THE CUSTOMIZATION SECTION
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
if optim == -1:
    initial_fitness = float("inf")  # for minimization problem
if optim == 1:
    initial_fitness = -float("inf")  # for maximization problem
# -----------------------------------------------------------------------------


class Particle:
    def __init__(self, bounds):
        self.particle_position = []  # particle position
        self.particle_velocity = []  # particle velocity
        self.local_best_particle_position = []  # best position of the particle
        self.fitness_local_best_particle_position = initial_fitness  # initial objective function value of the best particle position
        self.fitness_particle_position = initial_fitness  # objective function value of the particle position

        for i in range(nv):
            self.particle_position.append(
                random.uniform(bounds[i][0], bounds[i][1]))  # generate random initial position
            self.particle_velocity.append(random.uniform(-1, 1))  # generate random initial velocity

    def evaluate(self, build_model):
        self.fitness_particle_position = build_model(self.particle_position)
        # print(self.fitness_particle_position)

        if optim == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # update the local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best
        if optim == 1:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # update the local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best

    def update_velocity(self, global_best_particle_position):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()

            cognitive_velocity = c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w * self.particle_velocity[i] + cognitive_velocity + social_velocity

    def update_position(self, bounds):  # Using a periodic boundary handler https://pyswarms.readthedocs.io/en/latest/api/pyswarms.backend.html
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]

            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][0] + ((self.particle_position[i] - bounds[i][1]) % np.abs(bounds[i][0] - bounds[i][1]))

            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][1] - ((bounds[i][0] - self.particle_position[i]) % np.abs(bounds[i][0] - bounds[i][1]))

fitness_global_best_particle_position = initial_fitness
global_best_particle_position = []

# Build swarm
swarm = []
for i in range(particle_size):
    swarm.append(Particle(bounds))

fitness_history = []
param_history = []

tol = 1E-6
converg_count = 0

iter_count = 0
drift_limit = 2.5  # in percent

# for i in range(iterations):
while fitness_global_best_particle_position > drift_limit:
    print("Iteration {}".format(iter_count+1))

    # cycle through particles in swarm and evaluate fitness
    for j in range(particle_size):
        swarm[j].evaluate(build_model)

        # determine if current particle is the best (globally)
        if optim == -1:
            if swarm[j].fitness_particle_position < fitness_global_best_particle_position:
                global_best_particle_position = list(swarm[j].particle_position)
                fitness_global_best_particle_position = float(swarm[j].fitness_particle_position)
        if optim == 1:
            if swarm[j].fitness_particle_position > fitness_global_best_particle_position:
                global_best_particle_position = list(swarm[j].particle_position)
                fitness_global_best_particle_position = float(swarm[j].fitness_particle_position)

   # cycle through swarm and update velocities and position
    for j in range(particle_size):
        swarm[j].update_velocity(global_best_particle_position)
        swarm[j].update_position(bounds)

    fitness_history.append(fitness_global_best_particle_position)  # record the best fitness
    param_history.append(global_best_particle_position)  # record associated fitness parameters for each iteration

    print('iteration: {}, best_solution: {}, best_fitness: {}'.format(iter_count+1, global_best_particle_position,
                                                                      fitness_global_best_particle_position))

    if iter_count > 0:
        if abs(fitness_history[-1] - fitness_history[-2]) < tol:
            converg_count += 1
        else:
            converg_count = 0

        if converg_count == 20:
            break

    iter_count += 1

print('Optimal solution:', global_best_particle_position)
print('Objective function value:', fitness_global_best_particle_position)

run_time = time.time() - init_time
print("\nRun time:  {} secs".format(run_time))

convergence_history = open("./optimization_results/SMF_nzs_99.txt", 'w+')
convergence_history.write("Best Solution History: " + str(param_history) + "\n \n")
convergence_history.write("Best Fitness History: " + str(fitness_history) + "\n")
convergence_history.write("Run time: " + str(run_time) + " secs\n")
convergence_history.close()
# """
