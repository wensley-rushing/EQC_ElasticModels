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
# import opsvis as opsv
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Append directory of helper functions to Pyhton Path
sys.path.append('../')

from helper_functions.helper_funcs import refine_mesh
from helper_functions.helper_funcs import create_shell
from helper_functions.build_smf_column import create_columns
from helper_functions.build_smf_beam import create_beams
from helper_functions.get_beam_demands import process_beam_resp, compute_story_drifts
from helper_functions.get_beam_demands import cqc_modal_combo

# Define font properties for plot labels
mpl.rcParams['axes.edgecolor'] = 'grey'
mpl.rcParams['lines.markeredgewidth'] = 0.4
mpl.rcParams['lines.markeredgecolor'] = 'k'
plt.rcParams.update({'font.family': 'Times New Roman'})

axes_font = {'family': "sans-serif",
             'color': 'black',
             'size': 12
             }

title_font = {'family': 'sans-serif',
              'color': 'black',
              'weight': 'bold',
              'size': 16}

legend_font = {'family': 'Times New Roman',
              'size': 14}

# Clear memory
ops.wipe()

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

# Clear model  memory
ops.wipe()

# Model Builder
ops.model('basic', '-ndm', 3, '-ndf', 6)

# Create tag for shell element
nD_mattag = 1
plate_fiber_tag = 2
shell_sect_tag = 1

slab_thick = 165 * mm
fiber_thick = slab_thick / 3

shell_E =  26000 * MPa # Modulus of concrete
shell_nu = 0.2  # Poisson's ratio

ops.nDMaterial('ElasticIsotropic', nD_mattag, shell_E, shell_nu)
ops.nDMaterial('PlateFiber', plate_fiber_tag, nD_mattag)
ops.section('LayeredShell', shell_sect_tag, 3, plate_fiber_tag, fiber_thick, plate_fiber_tag, fiber_thick, plate_fiber_tag, fiber_thick)


# ============================================================================
# Beam properties
# ============================================================================
bm_nu = 0.28  # Poisson's ratio for steel

# The geometric properties of the columns will be defined using a W24x68 (metric W610x101)
bm_d = 602 * mm
bm_A = 13000 * mm**2
bm_E = 210 * GPa
bm_G = bm_E / (2*(1 + bm_nu))
bm_Iy = 29.3 * 1E6 * mm**4  # weak axis
bm_Iz = 762 * 1E6 * mm**4   # strong axis
bm_J = 778 * 1E3 * mm**4

bm_transf_tag_x = 3  # Beams oriented in Global-X direction
ops.geomTransf('PDelta', bm_transf_tag_x, 0, -1, 0)

bm_transf_tag_y = 4  # Beams oriented in Global-Y direction
ops.geomTransf('PDelta', bm_transf_tag_y, 1, 0, 0)  # -1, 0, 0

bm_section = [bm_A, bm_E, bm_G, bm_Iy, bm_Iz, bm_J, bm_transf_tag_x, bm_transf_tag_y]

# ============================================================================
# Column properties
# ============================================================================
col_nu = 0.28  # Poisson's ratio for steel
col_E = 210 * GPa

# The geometric properties of the columns for the East-West SMF will be defined using a W14x132 (metric W360x196)
col_d = 373 * mm
col_A_EW = 25000 * mm**2
col_G_EW = col_E / (2*(1 + col_nu))
col_Iy_EW = 228 * 1E6 * mm**4   # weak Iyy
col_Iz_EW = 637 * 1E6 * mm**4   # strong Ixx
col_J_EW = 5120 * 1E3 * mm**4

col_transf_tag_EW = 1
ops.geomTransf('PDelta', col_transf_tag_EW, 0, 1, 0)

col_EW_prop = [col_A_EW, col_G_EW, col_Iy_EW, col_Iz_EW, col_J_EW, col_E, col_transf_tag_EW]

# The geometric properties of the columns for the North-South SMF will be defined using a W14x132 (metric W360x196)
col_A_NS = 25000 * mm**2
col_G_NS = col_E / (2*(1 + col_nu))
col_Iy_NS = 637 * 1E6 * mm**4   # strong Ixx
col_Iz_NS = 228 * 1E6 * mm**4   # weak Iyy
col_J_NS = 5120 * 1E3 * mm**4

col_transf_tag_NS = 2
ops.geomTransf('PDelta', col_transf_tag_NS, 0, 1, 0)

col_NS_prop = [col_A_NS, col_G_NS, col_Iy_NS, col_Iz_NS, col_J_NS, col_E, col_transf_tag_NS]

# ============================================================================
# Define rigid material for beam-column joints elements in panel zone region
# ============================================================================
pz_d = bm_d / 2  # Half the depth of panel zone region
pz_w = col_d / 2 # Half the width of panel zone region

stiff_modif = 100

pz_A = col_A_NS * stiff_modif
pz_E = col_E * stiff_modif
pz_G = col_G_NS
pz_J = col_J_NS
pz_I = col_Iy_NS

rigid_mat_tag = 100
ops.uniaxialMaterial('Elastic', rigid_mat_tag, pz_E)

pzone_transf_tag_col = 100
pzone_transf_tag_bm_x = 200
pzone_transf_tag_bm_y = 300

ops.geomTransf('Linear', pzone_transf_tag_col, 0, 1, 0)
ops.geomTransf('Linear', pzone_transf_tag_bm_x, 0, -1, 0)
ops.geomTransf('Linear', pzone_transf_tag_bm_y, 1, 0, 0)

# ============================================================================
pz_prop_bm = [pz_w, pz_A, pz_E, pz_G, pz_J, pz_I, pzone_transf_tag_bm_x, pzone_transf_tag_bm_y]
pz_prop_col = [pz_A, pz_E, pz_G, pz_J, pz_I, pzone_transf_tag_col]

bm_prop = [bm_section, pz_prop_bm]
# ============================================================================

# Initialize dictionary to store node tags of COM for all floors
com_node_tags = {}

# ============================================================================
# Create floors
# ============================================================================

def create_floor(elev, floor_num, floor_label=''):

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
        create_columns(floor_num, smf_node_tags, col_EW_prop, col_NS_prop, pz_prop_col)
        create_beams(floor_num, elev, com_node, smf_node_tags, smf_coords_df, bm_prop)

    print('Floor ' + floor_num + ' created')



# Create pvd recorder
record_direc = './pvd/'
os.makedirs(record_direc, exist_ok=True)
ops.recorder('PVD', record_direc, '-precision', 3, '-dT', 1, *['mass', 'reaction'], 'eigen', 10)

print('Now creating SSMF model... \n')
create_floor(ground_flr, '00')
create_floor(flr1, '01', '1st')
create_floor(flr2, '02', '2nd')
create_floor(flr3, '03', '3rd')
create_floor(flr4, '04', '4th')
create_floor(flr5, '05', '5th')
create_floor(flr6, '06', '6th')
create_floor(flr7, '07', '7th')
create_floor(flr8, '08', '8th')
create_floor(flr9, '09', '9th')
create_floor(flr10, '10', '10th')
create_floor(roof_flr, '11', 'Roof')

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
# Nonlinear static pushover
# ============================================================================
push = 0

push_direc = 1 # Direction for pushover; 1: x-axis; 2: y-xis

if push:

    start_time = time.time()

    print('\nNow performing NSP... \n')
    max_drift = 2  # Maximum roof drift
    record_direc =  './pushover/'
    os.makedirs(record_direc, exist_ok=True)

    # Maintain constant gravity loads and reset time to zero
    ops.loadConst('-time', 0.0)
    ops.wipeAnalysis()

    # Assign lateral loads
    ts_tag = 10000
    pattern_tag = 10000

    ops.timeSeries('Constant', ts_tag)
    ops.pattern('Plain', pattern_tag, ts_tag)

    push_node_mass_x = modal_prop['totalMass'][0] * com_eigen_vec_x
    push_node_mass_y = modal_prop['totalMass'][0] * com_eigen_vec_y

    # ops.load(com_node_tags['01'], push_node_mass_x[0], 0., 0., 0., 0., 0.)
    # ops.load(com_node_tags['02'], push_node_mass_x[1], 0., 0., 0., 0., 0.)
    # ops.load(com_node_tags['03'], push_node_mass_x[2], 0., 0., 0., 0., 0.)
    # ops.load(com_node_tags['04'], push_node_mass_x[3], 0., 0., 0., 0., 0.)
    # ops.load(com_node_tags['05'], push_node_mass_x[4], 0., 0., 0., 0., 0.)
    # ops.load(com_node_tags['06'], push_node_mass_x[5], 0., 0., 0., 0., 0.)
    # ops.load(com_node_tags['07'], push_node_mass_x[6], 0., 0., 0., 0., 0.)
    # ops.load(com_node_tags['08'], push_node_mass_x[7], 0., 0., 0., 0., 0.)
    # ops.load(com_node_tags['09'], push_node_mass_x[8], 0., 0., 0., 0., 0.)
    # ops.load(com_node_tags['10'], push_node_mass_x[9], 0., 0., 0., 0., 0.)
    # ops.load(com_node_tags['11'], push_node_mass_x[10], 0., 0., 0., 0., 0.)

    ops.load(com_node_tags['01'], push_node_mass_x[0], push_node_mass_y[0], 0., 0., 0., 0.)
    ops.load(com_node_tags['02'], push_node_mass_x[1], push_node_mass_y[0], 0., 0., 0., 0.)
    ops.load(com_node_tags['03'], push_node_mass_x[2], push_node_mass_y[0], 0., 0., 0., 0.)
    ops.load(com_node_tags['04'], push_node_mass_x[3], push_node_mass_y[0], 0., 0., 0., 0.)
    ops.load(com_node_tags['05'], push_node_mass_x[4], push_node_mass_y[0], 0., 0., 0., 0.)
    ops.load(com_node_tags['06'], push_node_mass_x[5], push_node_mass_y[0], 0., 0., 0., 0.)
    ops.load(com_node_tags['07'], push_node_mass_x[6], push_node_mass_y[0], 0., 0., 0., 0.)
    ops.load(com_node_tags['08'], push_node_mass_x[7], push_node_mass_y[0], 0., 0., 0., 0.)
    ops.load(com_node_tags['09'], push_node_mass_x[8], push_node_mass_y[0], 0., 0., 0., 0.)
    ops.load(com_node_tags['10'], push_node_mass_x[9], push_node_mass_y[0], 0., 0., 0., 0.)
    ops.load(com_node_tags['11'], push_node_mass_x[10], push_node_mass_y[0], 0., 0., 0., 0.)

    # Recorder for base shear
    ops.recorder('Node', '-file', record_direc + 'baseShearX.txt', '-node', *smf_node_tags['00'].tolist(), '-dof', 1, 'reaction')
    ops.recorder('Node', '-file', record_direc + 'baseShearY.txt', '-node', *smf_node_tags['00'].tolist(), '-dof', 2, 'reaction')
    # ops.recorder('Node', '-file', record_direc + 'baseShearX_Y.txt', '-node', *smf_node_tags['00'].tolist(), '-dof', *[1, 2], 'reaction')

    # Recorder for COM displacement
    ops.recorder('Node', '-file', record_direc + 'COM_dispX.txt', '-node', *list(com_node_tags.values()), '-dof', 1, 'disp')
    ops.recorder('Node', '-file', record_direc + 'COM_dispY.txt', '-node', *list(com_node_tags.values()), '-dof', 2, 'disp')
    # ops.recorder('Node', '-file', record_direc + 'roof_COM_dispX_Y.txt', '-node', com_node_tags['11'], '-dof', *[1, 2], 'disp')


    'Displacemenet parameters'
    IDctrlNode = com_node_tags['11']   # Node where disp is read for disp control: COM of roof floor
    IDctrlDOF = push_direc             # Degree of freedom read for disp control (1 = x displacement)
    Dmax = max_drift / 100 * roof_flr  # Maximum displacement of pushover
    Dincr = 0.0002				       # Displacement increment
    tol = 1.0E-6  # Convergence tolerance for test

    'Analysis commands'
    ops.constraints('Transformation')  # Plain, Lagrange, Penalty, Transformation
    ops.numberer('Plain')

    ops.system('UmfPack')
    ops.test('NormDispIncr', tol, 100, 0)
    ops.algorithm('Newton')

    ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)
    ops.analysis('Static')
    Nsteps = math.floor(Dmax / Dincr)

    ok = 0
    for ii in range(Nsteps):

        # print('Step {} of {}'.format(ii, Nsteps))
        if ii % 100 == 0:
            print('Step {} of {}'.format(ii, Nsteps))

        ok = ops.analyze(1)

        if ok != 0:
            ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr/2)
            ok = ops.analyze(1)
            ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)

        if ok != 0:
            ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr/4)
            ok = ops.analyze(1)
            ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)

        if ok != 0:
            ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr/8)
            ok = ops.analyze(1)
            ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)

        if ok != 0:
            ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr/16)
            ok = ops.analyze(1)
            ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)

    end_time = time.time()
    total_time = end_time - start_time
    print("TIme taken to run NSP: {:.2f} secs.".format(total_time))

    # Shut down pushover recorders
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

    for ii in range (len(direcs)):

        # Create directory to save results
        mrsa_res_folder = './mrsa_results/dir' + axis[ii] + '/'
        os.makedirs(mrsa_res_folder, exist_ok=True)

        # Create recorder for beam-response in direction of excitation
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

        # Base shear
        ops.recorder('Node', '-file', mrsa_res_folder + 'baseShearX.txt', '-node', *smf_node_tags['00'].tolist(), '-dof', 1, 'reaction')
        ops.recorder('Node', '-file', mrsa_res_folder + 'baseShearY.txt', '-node', *smf_node_tags['00'].tolist(), '-dof', 2, 'reaction')

        # Recorder for COM displacement
        ops.recorder('Node', '-file', mrsa_res_folder + 'COM_dispX.txt', '-node', *list(com_node_tags.values()), '-dof', 1, 'disp')
        ops.recorder('Node', '-file', mrsa_res_folder + 'COM_dispY.txt', '-node', *list(com_node_tags.values()), '-dof', 2, 'disp')

        for jj in range(num_modes):
            ops.responseSpectrumAnalysis(direcs[ii], '-Tn', *spect_periods, '-Sa', *spect_acc, '-mode', jj + 1)

        # Shut down recorder for current direction of excitation
        ops.remove('recorders')


# Clear model
ops.wipe()


# ============================================================================
# Post-process MRSA results
# ============================================================================
bm_demands_X = process_beam_resp('./mrsa_results/dirX/', lambda_list, damping_ratio, num_modes)
bm_demands_Y = process_beam_resp('./mrsa_results/dirY/', lambda_list, damping_ratio, num_modes)

base_shearX = cqc_modal_combo(np.loadtxt('./mrsa_results/dirX/baseShearX.txt'), lambda_list, damping_ratio, num_modes).sum()

base_shearY = cqc_modal_combo(np.loadtxt('./mrsa_results/dirY/baseShearY.txt'), lambda_list, damping_ratio, num_modes).sum()

"""
# Deflection amplification factors
seismic_weight = 58036.35 * kN
ductility_fac = 4.0
kp  = 0.015 + 0.0075*(ductility_fac - 1)
kp = min(max(0.0015, kp), 0.03)

# pdelta_fac = kp * seismic_weight +

# Compute story drifts
# For MRSA in x-direction
com_dispX_x = np.loadtxt('./mrsa_results/dirX/COM_dispX.txt')
com_dispX_y = np.loadtxt('./mrsa_results/dirX/COM_dispY.txt')
story_driftX = compute_story_drifts(com_dispX_x, com_dispX_y, story_heights, lambda_list, damping_ratio, num_modes)

# For MRSA in x-direction
com_dispY_x = np.loadtxt('./mrsa_results/dirY/COM_dispX.txt')
com_dispY_y = np.loadtxt('./mrsa_results/dirY/COM_dispY.txt')
story_driftY = compute_story_drifts(com_dispY_x, com_dispY_y, story_heights, lambda_list, damping_ratio, num_modes)
"""
