# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:43:33 2023

@author: Uzo Uwaoma - udu@uw.edu
"""
# Import required modules
import os
import sys
import opensees.openseespy as ops
# import opsvis as opsv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter

# Append directory of helper functions to Pyhton Path
sys.path.append('../')

from helper_functions.create_floor_shell import refine_mesh
from helper_functions.create_floor_shell import create_shell
from helper_functions.cqc_modal_combo import modal_combo
from helper_functions.get_beam_col_demands import process_beam_col_resp
from helper_functions.get_story_drift import compute_story_drifts
from helper_functions.elf_new_zealand import nz_horiz_seismic_shear, nz_horiz_force_distribution
from helper_functions.get_spectral_shape_factor import spectral_shape_fac
from helper_functions.get_wall_reinforcement import get_wall_reinf_ratio
from helper_functions.get_dataframe_index import find_row

# Set plotting parameters
mpl.rcParams['axes.edgecolor'] = 'grey'
mpl.rcParams['lines.markeredgewidth'] = 0.4
mpl.rcParams['lines.markeredgecolor'] = 'k'
plt.rcParams.update({'font.family': 'Times New Roman'})

axes_font = {'family': "sans-serif",
              'color': 'black',
              'size': 8
              }

title_font = {'family': 'sans-serif',
              'color': 'black',
              'weight': 'bold',
              'size': 8}

legend_font = {'family': 'Times New Roman',
              'size': 8}


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

print('Base units are: \n Force: kN, \n Length: m, \n Time: sec. \n')

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

elev = [flr1, flr2, flr3, flr4, flr5, flr6, flr7, flr8, flr9, flr10, roof_flr]

# Generic material properties
conc_rho = 24 * kN / m**3 # Density of concrete
conc_fcp = 40 * MPa  # Compressive strength of concrete
steelE = 200 * GPa  # Modulus of steel

'''
Define array of wall sectional properties in meters
Each row specifies the length and thickness of each wall.
Rows are arranged from Wall 1 to Wall 10 as labelled in drawings.
'''
wall_prop = np.array([[7.550, .300],
                      [7.305, .300],
                      [7.640, .300],
                      [3.405, .300],
                      [7.475, .300],
                      [7.305, .300],
                      [4.895, .300],
                      [7.305, .300],
                      [7.400, .300],
                      [7.400, .300]]) * m

wall_prop_dict = {'wall1': wall_prop[0],
                  'wall2': wall_prop[1],
                  'wall3': wall_prop[2],
                  'wall4': wall_prop[3],
                  'wall5': wall_prop[4],
                  'wall6': wall_prop[5],
                  'wall7': wall_prop[6],
                  'wall8': wall_prop[7],
                  'wall9': wall_prop[8],
                  'wall10': wall_prop[9]}

# Create list of orientations of Wall 1 to Wall 10
wall_orient = ['NS', 'NS', 'EW', 'EW', 'NS',
               'NS', 'NS', 'EW', 'EW', 'EW']

# Column & wall centerline x-y coordinates in m
lfre_coords_dict = {'wall1':  [0, 3.625],
               'wall2': [21.210, 3.625],
               'wall3': [25.785, 5.825],
               'wall4': [16.635, 9.150],
               'wall5': [0, 12.888],
               'wall6': [13.010, 12.888],
               'wall7': [16.635, 12.888],  # 16.635, 11.818]
               'wall8': [16.635, 16.625],
               'wall9': [16.635, 31.025],
               'wall10':[25.785, 31.025],
               'col1': [6.505, 0],
               'col2': [13.010, 0],
               'col3': [29.410, 0],
               'col4': [6.505, 9.150],  # 6.505, 8.313
               'col5': [21.210, 9.150],
               'col6': [29.410, 9.150],
               'col7': [6.505, 16.625],
               'col8': [21.210, 16.625],
               'col9': [29.410, 16.625],
               'col10': [13.010, 23.825],
               'col11': [21.210, 23.825],
               'col12': [29.410, 23.825]}

'''
x & y coordinates of start and end points of all walls.
Will be used to define rigid elements at the top of the walls.

   _l =  left or bottom of wall, depending on orientation;
   _r =  right or top, depending on wall orientation
'''
wall_ends_dict = {'wall1_l': [0, 0],
                   'wall1_r': [0, 7.550],
                   'wall2_l': [21.210, 0],
                   'wall2_r': [21.210, 7.305],
                   'wall3_l': [21.770, 5.825],
                   'wall3_r': [29.410, 5.825],
                   'wall4_l': [14.9355, 9.150],
                   'wall4_r': [18.3375, 9.150],
                   'wall5_l': [0, 9.150],
                   'wall5_r': [0, 16.625],
                   'wall6_l': [13.010, 9.150],
                   'wall6_r': [13.010, 16.455],
                   'wall7_l': [16.635, 10.450],
                   'wall7_r': [16.635, 15.345],
                   'wall8_l': [12.9825, 16.625],
                   'wall8_r': [20.2875, 16.625],
                   'wall9_l': [13.010, 31.025],
                   'wall9_r': [20.410, 31.025],
                   'wall10_l': [22.010, 31.025],
                   'wall10_r': [29.410, 31.025]}


# Convert dictionaries to DataFrame
lfre_coords_df = pd.DataFrame.from_dict(lfre_coords_dict, orient='index', columns=['x', 'y'])
wall_ends_df = pd.DataFrame.from_dict(wall_ends_dict, orient='index', columns=['x', 'y'])

# Combine coordinates of center-nodes of walls and columns, and end-nodes of walls
# This will be used to define a mesh grid
lfre_node_positions = pd.concat([lfre_coords_df, wall_ends_df])

# Create a dataframe to store node tags nodes at walls/columns location.
lfre_node_tags = pd.DataFrame(columns=['00', '01', '02', '03', '04', '05',
                                       '06', '07', '08', '09', '10', '11'],
                              index=lfre_coords_df.index)

# Create a dataframe to store node tags nodes at the wall ends
wall_ends_node_tags = pd.DataFrame(columns=['00', '01', '02', '03', '04', '05',
                                            '06', '07', '08', '09', '10', '11'],
                                   index=wall_ends_df.index)


'Sort x and y-coordinates of LFRE. This will be used to define a mesh grid'
# Extract x & y coordinates, sort and remove dupllicates
col_wall_x_coords = sorted(list(set([coord for coord in lfre_node_positions['x']])))
col_wall_y_coords = sorted(list(set([coord for coord in lfre_node_positions['y']])))

col_wall_x_coords = np.array(list(col_wall_x_coords))
col_wall_y_coords = np.array(list(col_wall_y_coords))

# Create finer mesh
discretize = 0
if discretize:
    mesh_size = 1 * m  # Mesh size
    x_coords = refine_mesh(col_wall_x_coords, mesh_size)
    y_coords = refine_mesh(col_wall_y_coords, mesh_size)
else:
    x_coords = col_wall_x_coords
    y_coords = col_wall_y_coords


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
# Define shell properties for floor diaphragm
# ============================================================================
nD_mattag = 1
plate_fiber_tag = 2
shell_sect_tag = 1

slab_thick =  165 * mm
fiber_thick = slab_thick / 3

shell_E =  26000 * MPa # Modulus of concrete
shell_nu = 0.2  # Poisson's ratio

# ============================================================================
# Initialize dictionary to store node tags of COM for all floors
# Initialize dictionary to store total mass of each floor
# ============================================================================
com_node_tags = {}
total_floor_mass = {}


# ============================================================================
# Define function to create a floor
# ============================================================================

# Create floor
def create_floor(elev, floor_num, floor_label=''):

    node_compile = []  # Store node tags grouped according to their y-coordinates

    # Create nodes
    node_num = int(floor_num + '1000')

    # Create timeseries for assigning self weight of walls
    ts_tag = int(floor_num)
    pattern_tag = int(floor_num)

    ops.timeSeries('Constant', ts_tag)
    ops.pattern('Plain', pattern_tag, ts_tag)

    for jj in range(len(unique_ys)):
        x_vals = grouped_x_coord[jj]
        node_list = [] # Store all node tags for current floor

        for x_val in x_vals:

            # Check if the current node is at the location of a wall or column
            col_or_wall_node = (lfre_coords_df == [x_val, unique_ys[jj]]).all(1).any()

            if floor_num == '00':  # Only create bottom floor nodes at the location of columns/walls

                if col_or_wall_node:
                    ops.node(node_num, x_val, unique_ys[jj], elev)

                    # Assign Boundary conditions
                    # Wall: Fixed at base; Gravity Columns: Pinned at base
                    elem_type = find_row(lfre_coords_df, [x_val, unique_ys[jj]])  # Get the index/row name

                    if "wall" in elem_type:
                        ops.fix(node_num, 1, 1, 1, 1, 1, 1)

                    else:  # Gravity column
                        ops.fix(node_num, 1, 1, 1, 0, 0, 0)

            else:
                ops.node(node_num, x_val, unique_ys[jj], elev)

                # Pin gravity columns on upper floors
                if col_or_wall_node:
                    elem_type = find_row(lfre_coords_df, [x_val, unique_ys[jj]])  # Get the index/row name

                    if "col" in elem_type:
                        ops.fix(node_num, 1, 1, 1, 0, 0, 0)

            'Store node tags for nodes at the location of columns/walls'
            # Check if the current node is at the location of a wall or column
            if (lfre_coords_df == [x_val, unique_ys[jj]]).all(1).any():

                # Get the row index
                row_id = lfre_coords_df[(lfre_coords_df['x'] == x_val) & (lfre_coords_df['y'] == unique_ys[jj])].index.tolist()[0]

                # Assign node tag to `lfre_coords_df`
                lfre_node_tags.loc[row_id][floor_num] = node_num

                # Assign self weight of wall
                if 'wall' in row_id:
                    if floor_num != '00': # Wall weight is applied at the top so ground floor is excluded
                        if floor_num == '01':
                            wall_height = flr1  # 4.5m
                        else:
                            wall_height = typ_flr_height  #3.1m

                        # ==== Load
                        wall_self_weight = conc_rho * np.prod(wall_prop_dict[row_id]) * wall_height  # in kN
                        ops.load(node_num, 0, 0, -wall_self_weight, 0, 0, 0)

                        # ==== Mass
                        # First time mass is added, no need to retrieve existing mass
                        wall_mass = wall_self_weight / grav_metric  # in kN-s^2/m
                        neglig_mass = 1E-8 * kN * sec**2 / m
                        ops.mass(node_num, wall_mass, wall_mass, neglig_mass, neglig_mass, neglig_mass, neglig_mass)

            'Store node tags for nodes at the ends of wall section'
            # Check if the current node is at the end of a wall section
            if (wall_ends_df == [x_val, unique_ys[jj]]).all(1).any():

                # Get the row index
                row_id = wall_ends_df[(wall_ends_df['x'] == x_val) & (wall_ends_df['y'] == unique_ys[jj])].index.tolist()[0]

                # Assign node tag to `wall_ends_node_tags`
                wall_ends_node_tags.loc[row_id][floor_num] = node_num


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
    # Then create columns, walls, and wall rigid links
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

        com_node_tags[floor_num] = node_num

        # Create columns & walls
        create_columns(floor_num)
        create_walls(floor_num)
        create_wall_rigid_links(floor_num)

    print('Floor ' + floor_num + ' created')


# ============================================================================
# Create material properties for wall
# ============================================================================
wall_nu = 0.2  # Poisson's ratio for concrete
wall_E = 27898 * MPa  # Value taken from approved structural calcs
wall_G = wall_E / (2*(1 + wall_nu))
wall_transf_tag = 1  # Walls in EW & NS direction have the same transformation


def get_wall_prop(section_prop, orient):

    wall_length = section_prop[0]
    wall_thick = section_prop[1]

    wall_area = wall_length * wall_thick

    # Compute moment of inertias (using reduced sectional properties) based on wall orientation
    if orient == 'EW': # Wall orientation West-East
        wall_Iy = 0.5 * wall_length * (wall_thick**3) / 12
        wall_Iz = 0.5 *(wall_length**3) * wall_thick / 12

    else:  # Wall orientation is North-South
        wall_Iy = 0.5 * (wall_length**3) * wall_thick / 12
        wall_Iz = 0.5 * wall_length * (wall_thick**3) / 12

    return wall_area, wall_Iy, wall_Iz


def create_walls(floor_num):

    wall_tag = int('4' + floor_num + '01')  # 40101

    for ii in range(len(wall_prop)):
        wall_A, wall_Iy, wall_Iz = get_wall_prop(wall_prop[ii], wall_orient[ii])

        # element('ElasticTimoshenkoBeam', eleTag, *eleNodes, E_mod, G_mod, Area, Jxx, Iy, Iz, Avy, Avz, transfTag, <'-mass', massDens>, <'-cMass'>)
        ops.element('ElasticTimoshenkoBeam', wall_tag, lfre_node_tags.iloc[ii][floor_num] - 10000,
                    lfre_node_tags.iloc[ii][floor_num], wall_E, wall_G, wall_A, 0.0, wall_Iy, wall_Iz, wall_A, wall_A, wall_transf_tag)

        # Update reference to wall_tag
        wall_tag += 1


# ============================================================================
# Create material/geometric properties for wall rigid links
# ============================================================================
bm_nu = 0.28  # Poisson's ratio for steel
bm_d = 602 * mm
bm_A = 13000 * mm**2
bm_E = steelE
bm_G = bm_E / (2*(1 + bm_nu))
bm_I = 762 * 1E6 * mm**4   # strong axis
bm_J = 778 * 1E3 * mm**4

wall_link_A = bm_A * 100
wall_link_E = bm_E * 100
wall_link_G = bm_G
wall_link_J = bm_J
wall_link_I = bm_I

wall_link_transf_tag_x = 2 # Walls oriented in Global-X direction
wall_link_transf_tag_y = 3 # Walls oriented in Global-Y direction


def create_wall_rigid_links(floor_num):

    wall_rigid_tag = int('5' + floor_num + '01') # 50101

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

# ============================================================================
# Create columns
# ============================================================================
col_nu = 0.28  # Poisson's ratio for steel
col_E = steelE

# The geometric properties of the columns oriented in the East-West direction will be defined using a W14x132 (metric W360x196)
col_A_EW = 25000 * mm**2
col_G_EW = col_E / (2*(1 + col_nu))
col_Iy_EW = 228 * 1E6 * mm**4   # weak Iyy
col_Iz_EW = 637 * 1E6 * mm**4   # strong Ixx
col_J_EW = 5120 * 1E3 * mm**4

col_transf_tag = 4


# The geometric properties of the columns oriented in the North-South direction will be defined using a W14x132 (metric W360x196)
col_A_NS = 25000 * mm**2
col_G_NS = col_E / (2*(1 + col_nu))
col_Iy_NS = 637 * 1E6 * mm**4   # strong Ixx
col_Iz_NS = 228 * 1E6 * mm**4   # weak Iyy
col_J_NS = 5120 * 1E3 * mm**4


def create_columns(floor_num):

    col_tag = int('3' + floor_num + '01')  # 30101

    ops.element('elasticBeamColumn', col_tag, lfre_node_tags.loc['col1'][floor_num] - 10000, lfre_node_tags.loc['col1'][floor_num],
                col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 1

    ops.element('elasticBeamColumn', col_tag + 1, lfre_node_tags.loc['col2'][floor_num] - 10000, lfre_node_tags.loc['col2'][floor_num],
                col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 2

    ops.element('elasticBeamColumn', col_tag + 2, lfre_node_tags.loc['col3'][floor_num] - 10000, lfre_node_tags.loc['col3'][floor_num],
                col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 3

    ops.element('elasticBeamColumn', col_tag + 3, lfre_node_tags.loc['col4'][floor_num] - 10000, lfre_node_tags.loc['col4'][floor_num],
                col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 4

    ops.element('elasticBeamColumn', col_tag + 4, lfre_node_tags.loc['col5'][floor_num] - 10000, lfre_node_tags.loc['col5'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 5

    ops.element('elasticBeamColumn', col_tag + 5, lfre_node_tags.loc['col6'][floor_num] - 10000, lfre_node_tags.loc['col6'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 6

    ops.element('elasticBeamColumn', col_tag + 6, lfre_node_tags.loc['col7'][floor_num] - 10000, lfre_node_tags.loc['col7'][floor_num],
                col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 7

    ops.element('elasticBeamColumn', col_tag + 7, lfre_node_tags.loc['col8'][floor_num] - 10000, lfre_node_tags.loc['col8'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 8

    ops.element('elasticBeamColumn', col_tag + 8, lfre_node_tags.loc['col9'][floor_num] - 10000, lfre_node_tags.loc['col9'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 9

    ops.element('elasticBeamColumn', col_tag + 9, lfre_node_tags.loc['col10'][floor_num] - 10000, lfre_node_tags.loc['col10'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 10

    ops.element('elasticBeamColumn', col_tag + 10, lfre_node_tags.loc['col11'][floor_num] - 10000, lfre_node_tags.loc['col11'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 11

    ops.element('elasticBeamColumn', col_tag + 11, lfre_node_tags.loc['col12'][floor_num] - 10000, lfre_node_tags.loc['col12'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 12


# ============================================================================
# Model builder
# ============================================================================
def build_model():
    # Clear model  memory
    ops.wipe()

    # Model Builder
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # Create shell material for floor diaphragm
    ops.nDMaterial('ElasticIsotropic', nD_mattag, shell_E, shell_nu)
    ops.nDMaterial('PlateFiber', plate_fiber_tag, nD_mattag)
    ops.section('LayeredShell', shell_sect_tag, 3, plate_fiber_tag, fiber_thick, plate_fiber_tag, fiber_thick, plate_fiber_tag, fiber_thick)

    # Define geometric transformation for walls
    ops.geomTransf('Linear', wall_transf_tag, 0, 1, 0)

    # Define geometric transformation for rigid wall links
    ops.geomTransf('Linear', wall_link_transf_tag_x, 0, -1, 0)
    ops.geomTransf('Linear', wall_link_transf_tag_y, 1, 0, 0)  # -1, 0, 0

    # Define geometric transformation for columns
    ops.geomTransf('Linear', col_transf_tag, 0, 1, 0)  # Same Geometric transformation applies to columns in both directions

    # Create all floors of building
    print('Now creating RCSW model... \n')
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
    # Create regions for steel columns & RC Walls based on floor
    # ============================================================================
    # Get all element tags
    elem_tags = ops.getEleTags()

    floor_nums = ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11']

    col_tags = []
    wall_tags = []

    for floor in floor_nums:
        floor_col_tags = []
        floor_wall_tags = []

        for tag in elem_tags:

            # Only select column elements
            if str(tag).startswith('3' + floor):
                floor_col_tags.append(tag)

            # Only select wall elements
            elif str(tag).startswith('4' + floor):
                floor_wall_tags.append(tag)

        col_tags.append(floor_col_tags)
        wall_tags.append(floor_wall_tags)

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

    # Walls
    ops.region(401, '-eleOnly', *wall_tags[0])  # Region for all walls on 1st floor
    ops.region(402, '-eleOnly', *wall_tags[1])  # Region for all walls on 2nd floor
    ops.region(403, '-eleOnly', *wall_tags[2])  # Region for all walls on 3rd floor
    ops.region(404, '-eleOnly', *wall_tags[3])  # Region for all walls on 4th floor
    ops.region(405, '-eleOnly', *wall_tags[4])  # Region for all walls on 5th floor
    ops.region(406, '-eleOnly', *wall_tags[5])  # Region for all walls on 6th floor
    ops.region(407, '-eleOnly', *wall_tags[6])  # Region for all walls on 7th floor
    ops.region(408, '-eleOnly', *wall_tags[7])  # Region for all walls on 8th floor
    ops.region(409, '-eleOnly', *wall_tags[8])  # Region for all walls on 9th floor
    ops.region(410, '-eleOnly', *wall_tags[9])  # Region for all walls on 10th floor
    ops.region(411, '-eleOnly', *wall_tags[10]) # Region for all walls on 11th floor


# Generate model
build_model()

# Create pvd recorder
record_direc = './pvd/'
os.makedirs(record_direc, exist_ok=True)
ops.recorder('PVD', record_direc, '-precision', 3, '-dT', 1, *['mass', 'eigen', 10])

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

    modal_prop = ops.modalProperties('-file', 'ModalReport_RCSW.txt', '-unorm', '-return')

    # Apply Damping
    damping_ratio = 0.05  # 5% Damping

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

ops.recorder('Element', '-file', grav_direc + 'colFx.txt', '-ele', 30110, 30111, 30112, '-dof', 1,2, 3, 4, 5, 6, 'globalForce')
ops.recorder('Node', '-file', grav_direc + 'nodeRxn.txt', '-node', *lfre_node_tags['00'].tolist(), '-dof', 1, 2, 3, 4, 5, 6, 'reaction')

# Recorders to check correct implementation of boundary condtions for gravity columns
# ops.recorder('Element', '-file', grav_direc + 'floor01_colResp.txt', '-region', 301, 'force')
# ops.recorder('Element', '-file', grav_direc + 'floor02_colResp.txt', '-region', 302, 'force')
# ops.recorder('Element', '-file', grav_direc + 'floor03_colResp.txt', '-region', 303, 'force')
# ops.recorder('Element', '-file', grav_direc + 'floor04_colResp.txt', '-region', 304, 'force')
# ops.recorder('Element', '-file', grav_direc + 'floor05_colResp.txt', '-region', 305, 'force')
# ops.recorder('Element', '-file', grav_direc + 'floor06_colResp.txt', '-region', 306, 'force')
# ops.recorder('Element', '-file', grav_direc + 'floor07_colResp.txt', '-region', 307, 'force')
# ops.recorder('Element', '-file', grav_direc + 'floor08_colResp.txt', '-region', 308, 'force')
# ops.recorder('Element', '-file', grav_direc + 'floor09_colResp.txt', '-region', 309, 'force')
# ops.recorder('Element', '-file', grav_direc + 'floor10_colResp.txt', '-region', 310, 'force')
# ops.recorder('Element', '-file', grav_direc + 'floor11_colResp.txt', '-region', 311, 'force')


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
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor01_wallResp.txt', '-precision', 9, '-region', 401, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor02_wallResp.txt', '-precision', 9, '-region', 402, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor03_wallResp.txt', '-precision', 9, '-region', 403, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor04_wallResp.txt', '-precision', 9, '-region', 404, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor05_wallResp.txt', '-precision', 9, '-region', 405, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor06_wallResp.txt', '-precision', 9, '-region', 406, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor07_wallResp.txt', '-precision', 9, '-region', 407, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor08_wallResp.txt', '-precision', 9, '-region', 408, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor09_wallResp.txt', '-precision', 9, '-region', 409, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor10_wallResp.txt', '-precision', 9, '-region', 410, 'force')
        ops.recorder('Element', '-file', mrsa_res_folder + 'floor11_wallResp.txt', '-precision', 9, '-region', 411, 'force')

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

        # Recorders for COM displacement
        ops.recorder('Node', '-file', mrsa_res_folder + 'COM_disp' + axis[ii] + '.txt',
                     '-node', *list(com_node_tags.values()), '-dof', direcs[ii], 'disp')

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
lower_left_corner_dispX = modal_combo(np.loadtxt('./mrsa_results/dirX/lowerLeftCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
upper_right_corner_dispX = modal_combo(np.loadtxt('./mrsa_results/dirX/upperRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
lower_right_corner_dispX = modal_combo(np.loadtxt('./mrsa_results/dirX/lowerRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)

tir_x_edgeE = np.maximum(upper_right_corner_dispX, lower_right_corner_dispX) / (0.5*(upper_right_corner_dispX + lower_right_corner_dispX))  # Right edge of building plan
tir_x_edgeF = np.maximum(lower_left_corner_dispX, lower_right_corner_dispX) / (0.5*(lower_left_corner_dispX + lower_right_corner_dispX))    # Bottom edge of building plan

# ===== MRSA - Y
lower_left_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/lowerLeftCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
upper_right_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/upperRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)
lower_right_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/lowerRightCornerDisp.txt'), lambda_list, damping_ratio, num_modes)

tir_y_edgeE = np.maximum(upper_right_corner_dispY, lower_right_corner_dispY) / (0.5*(upper_right_corner_dispY + lower_right_corner_dispY))  # Right edge of building plan
tir_y_edgeF = np.maximum(lower_left_corner_dispY, lower_right_corner_dispY) / (0.5*(lower_left_corner_dispY + lower_right_corner_dispY))    # Bottom edge of building plan

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
ductility_factor = 1.25  # RCSW
story_weights = np.array(list(total_floor_mass.values())) * grav_metric
seismic_weight = story_weights.sum()

elf_base_shear = nz_horiz_seismic_shear(spectral_shape_factor, hazard_factor,
                                        return_per_factor_sls, return_per_factor_uls,
                                        fault_factor, perform_factor, ductility_factor,
                                        seismic_weight)

elf_force_distrib = nz_horiz_force_distribution(elf_base_shear, story_weights,
                                                np.cumsum(story_heights))

# Compute factors for scaling MRSA demands to ELF demands
elf_mrsaX_scale_factor = max(elf_base_shear / mrsa_base_shearX, 1)
elf_mrsaY_scale_factor = max(elf_base_shear / mrsa_base_shearY, 1)


# ============================================================================
# Check drift and stability requirements
# ============================================================================

def check_drift_and_stability(drift_X_dir, drift_Y_dir):

    # CHECK DRIFT REQUIREMENTS
    max_story_drift = max(drift_X_dir.max(), drift_Y_dir.max())
    drift_ok = max_story_drift < 2.5  # Maximum story drift limit = 2.5%  NZS 1170.5:2004 - Sect 7.5.1

    print('\nMaximum story drift: {:.2f}%'.format(max_story_drift))
    if drift_ok:
        print('Story drift requirements satisfied.')
    else:
        print('Story drift requirements NOT satisfied.')

    # CHECK STABILITY REQUIREMENTS (P-DELTA) NZS 1170.5:2004 - Sect 6.5.1
    thetaX = story_weights * 0.01 * drift_X_dir / (elf_force_distrib * story_heights)
    thetaY = story_weights * 0.01 * drift_Y_dir / (elf_force_distrib * story_heights)

    max_theta = max(thetaX.max(), thetaY.max())
    theta_ok = max_theta < 0.3

    print('\nMaximum stability coefficient: {:.2f}'.format(max_theta))
    if theta_ok:
        print('Stability requirements satisfied.')
    else:
        print('Stability requirements NOT satisfied.')


# Load in COM displacements from MRSA
mrsa_com_dispX = np.loadtxt('./mrsa_results/dirX/COM_dispX.txt')  # For MRSA in x-direction
mrsa_com_dispY = np.loadtxt('./mrsa_results/dirY/COM_dispY.txt')  # For MRSA in y-direction

# Drift amplification factor
drift_modif_fac = 1.5  # NZS 1170.5-2004: Table 7.1

pdelta_method = "B"

if pdelta_method == "A":  # (NZS 1170.5:2004 - Sect. 6.5.4.1)

    # Compute story drifts
    story_driftX = compute_story_drifts(mrsa_com_dispX, story_heights, lambda_list, damping_ratio, num_modes)
    story_driftY = compute_story_drifts(mrsa_com_dispY, story_heights, lambda_list, damping_ratio, num_modes)

    # Scale drifts by elf-to-mrsa base shear factor # NZS 1170.5-2004: Sect 5.2.2.2b
    story_driftX *= elf_mrsaX_scale_factor
    story_driftY *= elf_mrsaY_scale_factor

    kp  = 0.015 + 0.0075*(ductility_factor - 1)
    kp = min(max(0.0015, kp), 0.03)
    pdelta_fac = (kp * seismic_weight + elf_base_shear) / elf_base_shear  # NZS 1170.5-2004: Sec 7.2.1.2 & 6.5.4.1
    # pdelta_fac = 1

    # Amplify drifts by required factors
    story_driftX *=  (ductility_factor * pdelta_fac * drift_modif_fac)
    story_driftY *=  (ductility_factor * pdelta_fac * drift_modif_fac)

    check_drift_and_stability(story_driftX, story_driftY)

else: # Method B (NZS 1170.5:2004 - Sect. 6.5.4.2 & Commentary Sect. C6.5.4.2)

    # Modal combination on peak COM displacements from MRSA
    mrsa_total_com_dispX = modal_combo(mrsa_com_dispX, lambda_list, damping_ratio, num_modes)
    mrsa_total_com_dispY = modal_combo(mrsa_com_dispY, lambda_list, damping_ratio, num_modes)

    # Scale COM displacements by elf-to-mrsa base shear factor # NZS 1170.5-2004: Sect 5.2.2.2b
    mrsa_total_com_dispX *= elf_mrsaX_scale_factor
    mrsa_total_com_dispY *= elf_mrsaY_scale_factor

    # Amplify COM displacements by ductility factor
    # NZS 1170.5:2004 Commentary Sect. C6.5.4.2 Step 2
    mrsa_total_com_dispX *= ductility_factor
    mrsa_total_com_dispY *= ductility_factor

    # Compute interstory displacements
    inter_story_dispX = np.insert(np.diff(mrsa_total_com_dispX), 0, mrsa_total_com_dispX[0])
    inter_story_dispY = np.insert(np.diff(mrsa_total_com_dispY), 0, mrsa_total_com_dispY[0])

    # Compute story shear force due to PDelta actions
    # NZS 1170.5:2004 Commentary Sect. C6.5.4.2 Step 3a
    story_shear_forceX  = story_weights * inter_story_dispX / story_heights
    story_shear_forceY  = story_weights * inter_story_dispY / story_heights

    # Compute lateral forces to be used in static analysis for PDelta effects
    # NZS 1170.5:2004 Commentary Sect. C6.5.4.2 Step 3b
    lateral_forces_pDeltaX = np.insert(np.diff(story_shear_forceX), 0, story_shear_forceX[0])
    lateral_forces_pDeltaY = np.insert(np.diff(story_shear_forceY), 0, story_shear_forceY[0])


# ===================================================================================================
# Perform static analysis for accidental torsional moment & PDelta effects - method B (if applicable)
# ===================================================================================================
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

        # Loop through each COM node and apply torsional moment & PDelta lateral force if applicable
        for kk in range(len(com_nodes)):
            if torsional_direc[ii] == 'X' and pdelta_method == "A":  # Only torsional moment is applied about z-axis
                ops.load(com_nodes[kk], 0., 0., 0., 0., 0., torsional_mom_x[kk] * torsional_sign[jj])

            elif torsional_direc[ii] == 'X' and pdelta_method == "B": # Torsional moment about z-axis & PDelta "Method B" forces are applied
                ops.load(com_nodes[kk], lateral_forces_pDeltaX[kk], 0., 0., 0., 0., torsional_mom_x[kk] * torsional_sign[jj])

            elif torsional_direc[ii] == 'Y' and pdelta_method == "A":  # Only torsional moment is applied about z-axis
                ops.load(com_nodes[kk], 0., 0., 0., 0., 0., torsional_mom_y[kk] * torsional_sign[jj])

            elif torsional_direc[ii] == 'Y' and pdelta_method == "B":  # Torsional moment about z-axis & PDelta "Method B" forces are applied
                ops.load(com_nodes[kk], 0., lateral_forces_pDeltaY[kk], 0., 0., 0., torsional_mom_y[kk] * torsional_sign[jj])

        # Create directory to save results
        accident_torsion_res_folder = './accidental_torsion_results/' + torsional_folder[jj] + torsional_direc[ii] + '/'
        os.makedirs(accident_torsion_res_folder, exist_ok=True)

        # Create recorders for column response in direction of excitation
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

        # Create recorders for wall response in direction of excitation
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor01_wallResp.txt', '-precision', 9, '-region', 401, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor02_wallResp.txt', '-precision', 9, '-region', 402, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor03_wallResp.txt', '-precision', 9, '-region', 403, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor04_wallResp.txt', '-precision', 9, '-region', 404, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor05_wallResp.txt', '-precision', 9, '-region', 405, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor06_wallResp.txt', '-precision', 9, '-region', 406, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor07_wallResp.txt', '-precision', 9, '-region', 407, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor08_wallResp.txt', '-precision', 9, '-region', 408, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor09_wallResp.txt', '-precision', 9, '-region', 409, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor10_wallResp.txt', '-precision', 9, '-region', 410, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor11_wallResp.txt', '-precision', 9, '-region', 411, 'force')

        # Recorders for COM displacement
        ops.recorder('Node', '-file', accident_torsion_res_folder + 'COM_disp' + torsional_direc[ii] + '.txt', '-node', *list(com_node_tags.values()), '-dof', elf_dof[ii], 'disp')

        # Base shear
        ops.recorder('Node', '-file', accident_torsion_res_folder + 'baseShear' + torsional_direc[ii] + '.txt', '-node',
                      *lfre_node_tags['00'].tolist(), '-dof', elf_dof[ii], 'reaction')  # Fx or Fy

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

if pdelta_method == "B":
    # Process drifts due to PDelta lateral forces
    pdelta_com_disp_posX = np.loadtxt('./accidental_torsion_results/positiveX/COM_dispX.txt')
    pdelta_com_disp_negX = np.loadtxt('./accidental_torsion_results/negativeX/COM_dispX.txt')
    pdelta_com_disp_posY = np.loadtxt('./accidental_torsion_results/positiveY/COM_dispY.txt')
    pdelta_com_disp_negY = np.loadtxt('./accidental_torsion_results/negativeY/COM_dispY.txt')

    pdelta_com_dispX = np.maximum(pdelta_com_disp_posX, pdelta_com_disp_negX)
    pdelta_com_dispY = np.maximum(pdelta_com_disp_posY, pdelta_com_disp_negY)

    # Determine subsoil factor NZS 1170.5:2004 Sect. C6.5.4.2 Step 4
    # Case study building is in site subclass C.
    if periods[0] < 2.0:
        subsoil_factor_K = 1.0
    elif 2.0 <= periods[0] <= 4.0:
        subsoil_factor_K = (6 - periods[0]) / 4
    else:
        subsoil_factor_K = 4


    if ductility_factor <= 3.5:
        subsoil_factor_beta = 2 * ductility_factor * subsoil_factor_K / 3.5
    else:
        subsoil_factor_beta = 2 * subsoil_factor_K

    subsoil_factor_beta = max(subsoil_factor_beta, 1.0)

    # When using method B, element demands need to be scaled up by subsoil_factor_beta
    pdelta_fac = subsoil_factor_beta

    # Amplify PDelta COM displacements by subsoil_factor_beta and ductility factor
    pdelta_com_dispX *= (subsoil_factor_beta * ductility_factor)
    pdelta_com_dispY *= (subsoil_factor_beta * ductility_factor)

    # Add up COM displacements fropm MRSA & PDelta checks
    total_com_dispX = mrsa_total_com_dispX + pdelta_com_dispX
    total_com_dispY = mrsa_total_com_dispY + pdelta_com_dispY

    # Compute total interstory displacements
    total_inter_story_dispX = np.insert(np.diff(total_com_dispX), 0, total_com_dispX[0])
    total_inter_story_dispY = np.insert(np.diff(total_com_dispY), 0, total_com_dispY[0])

    # Compute story drift ratios
    story_driftX  = total_inter_story_dispX / story_heights * 100
    story_driftY  = total_inter_story_dispY / story_heights * 100

    # Amplify story drift ration by drift factor
    story_driftX *= drift_modif_fac
    story_driftY *= drift_modif_fac

    check_drift_and_stability(story_driftX, story_driftY)


# CHECK STRENGTH REQUIREMENTS

# Save story drifts
np.savetxt('driftX-PDeltaMethod{}.txt'.format(pdelta_method), story_driftX, fmt='%.2f')
np.savetxt('driftY-PDeltaMethod{}.txt'.format(pdelta_method), story_driftY, fmt='%.2f')

# ============================================================================
# Post-process MRSA & accidental torsion results
# ============================================================================
col_demands_X = process_beam_col_resp('col', './mrsa_results/dirX/', './accidental_torsion_results/positiveX/',
                                      './accidental_torsion_results/negativeX/', lambda_list, damping_ratio,
                                      num_modes, elf_mrsaX_scale_factor, pdelta_fac)

col_demands_Y = process_beam_col_resp('col', './mrsa_results/dirY/', './accidental_torsion_results/positiveY/',
                                      './accidental_torsion_results/negativeY/', lambda_list, damping_ratio,
                                      num_modes, elf_mrsaY_scale_factor, pdelta_fac)

wall_demands_X, wall_axialLoad_X, wall_mom_X = process_beam_col_resp('wall', './mrsa_results/dirX/', './accidental_torsion_results/positiveX/',
                                      './accidental_torsion_results/negativeX/', lambda_list, damping_ratio,
                                      num_modes, elf_mrsaX_scale_factor, pdelta_fac)

wall_demands_Y, wall_axialLoad_Y, wall_mom_Y = process_beam_col_resp('wall', './mrsa_results/dirY/', './accidental_torsion_results/positiveY/',
                                      './accidental_torsion_results/negativeY/', lambda_list, damping_ratio,
                                      num_modes, elf_mrsaY_scale_factor, pdelta_fac)

# Compute wall axial load ratios
wall_sect_area = wall_prop[:, 0] * wall_prop[:, 1]
wall_alr_X = (wall_axialLoad_X.divide(wall_sect_area * conc_fcp, axis='columns') * 100).round(2)
wall_alr_Y = (wall_axialLoad_Y.divide(wall_sect_area * conc_fcp, axis='columns') * 100).round(2)

# Base shear due to static accidental torsion analysis
accid_torsion_baseShear_pos_X = np.loadtxt('./accidental_torsion_results/positiveX/baseShearX.txt').sum()
accid_torsion_baseShear_neg_X = np.loadtxt('./accidental_torsion_results/negativeX/baseShearX.txt').sum()

accid_torsion_baseShear_pos_Y = np.loadtxt('./accidental_torsion_results/positiveY/baseShearY.txt').sum()
accid_torsion_baseShear_neg_Y = np.loadtxt('./accidental_torsion_results/negativeY/baseShearY.txt').sum()

base_shearX = max((mrsa_base_shearX + accid_torsion_baseShear_pos_X), (mrsa_base_shearX + accid_torsion_baseShear_neg_X))
base_shearY = max((mrsa_base_shearY + accid_torsion_baseShear_pos_Y), (mrsa_base_shearY + accid_torsion_baseShear_neg_Y))

# Generate story drift plots
fig, ax = plt.subplots(1, 2, figsize=(6.0, 7.5), sharex=True, sharey=True, constrained_layout=True)
fig.suptitle('Story drift ratios - PDelta Method {}'.format(pdelta_method), fontdict=title_font)

ax[0].vlines(story_driftX[0], 0.0, elev[0])
ax[1].vlines(story_driftY[0], 0.0, elev[0])

for ii in range(1, len(story_driftX)):
    ax[0].hlines(elev[ii-1], story_driftX[ii-1], story_driftX[ii])
    ax[0].vlines(story_driftX[ii],  elev[ii-1], elev[ii])

    ax[1].hlines(elev[ii-1], story_driftY[ii-1], story_driftY[ii])  # Correct
    ax[1].vlines(story_driftY[ii],  elev[ii-1], elev[ii])


ax[0].set_title('X - Direction', fontsize=12, family='Times New Roman')
ax[1].set_title('Y- Direction', fontsize=12, family='Times New Roman')

ax[0].set_ylabel('Story elevation (m)', fontdict=axes_font)

for axx in ax.flat:
    axx.set_xlim(0.0)
    axx.set_ylim(0.0, elev[-1])

    axx.grid(True, which='major', axis='both', ls='-.', linewidth=0.6)

    axx.set_yticks(elev)

    axx.set_xlabel('Story drift ratio (%)', fontdict=axes_font)

    axx.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    axx.tick_params(axis='both', direction='in', colors='grey', labelcolor='grey', zorder=3.0, labelsize=8.0)

# plt.savefig('DriftPlots-PDeltaMethod{}.png'.format(pdelta_method), dpi=1200)


# Compute wall reinforcing ratios
steel_fy = 500 * MPa
wall_reinf_ratio = pd.DataFrame(index=wall_mom_X.index, columns=wall_mom_X.columns)

wall_index = 0

for col in wall_mom_X:

    num_floors = wall_mom_X.shape[0]
    reinf_ratio = np.zeros(num_floors)

    wall_mom = wall_mom_X[col].to_numpy()
    wall_load = wall_axialLoad_X[col].to_numpy()

    for ii in range(num_floors):
        reinf_ratio[ii] = get_wall_reinf_ratio(wall_prop[wall_index][0], wall_prop[wall_index][1],
                             wall_load[ii], wall_mom[ii], conc_fcp, steel_fy, steelE)

    wall_reinf_ratio[col] = reinf_ratio
    wall_index += 1


# wall_prop = np.array([[7.550, .300],
#                       [7.305, .300],
#                       [7.640, .300],
#                       [3.405, .300],
#                       [7.475, .300],
#                       [7.305, .300],
#                       [4.895, .300],
#                       [7.305, .300],
#                       [7.400, .300],
#                       [7.400, .300]]) * m

# wall_ends_dict = {'wall1_l': [0, 0],
#                   'wall1_r': [0, wall_prop[0][0]],
#                   'wall2_l': [21.210, 0],
#                   'wall2_r': [21.210, wall_prop[1][0]],
#                   'wall3_l': [21.770, 5.825],
#                   'wall3_r': [21.770 + wall_prop[2][0], 5.825],
#                   'wall4_l': [14.9355, 9.150],
#                   'wall4_r': [14.9355 + wall_prop[3][0], 9.150],
#                   'wall5_l': [0, 9.150],
#                   'wall5_r': [0, 9.150 + wall_prop[4][0]],
#                   'wall6_l': [13.010, 9.150],
#                   'wall6_r': [13.010, 9.150 + wall_prop[5][0]],
#                   'wall7_l': [16.635, 10.450],
#                   'wall7_r': [16.635, 10.450 + wall_prop[6][0]],
#                   'wall8_l': [12.9825, 16.625],
#                   'wall8_r': [12.9825 + wall_prop[7][0], 16.625],
#                   'wall9_l': [13.010, 31.025],
#                   'wall9_r': [13.010 + wall_prop[8][0], 31.025],
#                   'wall10_l': [22.010, 31.025],
#                   'wall10_r': [22.010 + wall_prop[9][0], 31.025]}
