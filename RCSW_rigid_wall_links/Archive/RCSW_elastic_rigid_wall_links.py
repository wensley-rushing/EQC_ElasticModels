# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:43:33 2023

@author: Uzo Uwaoma - udu@uw.edu
"""
# Import required modules
import os
import sys
import math
import openseespy.opensees as ops
# import opsvis as opsv
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Append directory of helper functions to Pyhton Path
sys.path.append('../')

from helper_functions.create_floor_shell import refine_mesh
from helper_functions.create_floor_shell import create_shell


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

# Generic material properties
conc_rho = 24 * kN / m**3 # Density of concrete

# Column & wall centerline x-y coordinates in m
lfre_coords_dict = {'wall1':  [0, 3.625],
               'wall2':  [21.210, 3.625],
               'wall3':  [25.785, 5.825],
               'wall4':  [16.635, 9.150],
               'wall5':  [0, 12.888],
               'wall6':  [13.010, 12.888],
               'wall7':  [16.635, 12.888],  # 16.635, 11.818]
               'wall8':  [16.635, 16.625],
               'wall9':  [16.635, 31.025],
               'wall10':  [25.785, 31.025],
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
    mesh_size = 4 * m  # Mesh size - 4m x 4m elements
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

'''
Define array of wall sectional properties in meters
Each row specifies the length and thickness of each wall.
Rows are arranged from Wall 1 to Wall 10 as labelled in drawings.
'''
wall_prop = np.array([[7.550, .300],
                      [7.305, .300],
                      [7.775, .300],
                      [3.405, .400],
                      [7.550, .300],
                      [7.405, .300],
                      [4.895, .300],
                      [7.305, .300],
                      [7.550, .300],
                      [7.550, .300]]) * m

wall_prop_dict = {'wall1':  [7.550, .300],
               'wall2':  [7.305, .300],
               'wall3':  [7.775, .300],
               'wall4':  [3.405, .300],
               'wall5':  [7.550, .300],
               'wall6':  [7.405, .300],
               'wall7':  [4.895, .300],
               'wall8':  [7.305, .300],
               'wall9':  [7.550, .300],
               'wall10':  [7.550, .300]}

# Create list of orientations of Wall 1 to Wall 10
wall_orient = ['NS', 'NS', 'EW', 'EW', 'NS',
               'NS', 'NS', 'EW', 'EW', 'EW']


# Initialize dictionary to store node tags of COM for all floors
com_node_tags = {}

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
# Main model builder
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

            if floor_num == '00':
                # Only create nodes at the locations of columns and walls

                # Check if the current node is at the location of a wall or column
                if (lfre_coords_df == [x_val, unique_ys[jj]]).all(1).any():

                    ops.node(node_num, x_val, unique_ys[jj], elev)

                    # Assign Boundary conditions
                    ops.fix(node_num, 1, 1, 1, 1, 1, 1)

            else:
                ops.node(node_num, x_val, unique_ys[jj], elev)

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

                # Assign node tag to `lfre_coords_df`
                wall_ends_node_tags.loc[row_id][floor_num] = node_num


            # Move to next node
            node_list.append(node_num)
            node_num += 1

        node_compile.append(node_list)

    # Create additional node for COM
    # Get all node tags in current floor
    floor_node_tags = [node for node_list in node_compile for node in node_list]

    if floor_num != '00':
        ops.node(node_num, 16.363, 14.060, elev)

        # Impose rigid diaphragm constraint
        ops.rigidDiaphragm(3, node_num, *floor_node_tags)

        # Constraints for Rigid Diaphragm Primary node
        ops.fix(node_num, 0, 0, 1, 1, 1, 0)  # dx, dy, dz, rx, ry, rz

        com_node_tags[floor_num] = node_num

    # Create shells, columns & walls
    if floor_num != '00':
        create_shell(floor_num, node_compile, shell_sect_tag, num_y_groups)
        create_columns(floor_num)
        create_walls(floor_num)
        create_wall_rigid_links(floor_num)

    print('Floor ' + floor_num + ' created')


# ============================================================================
# Create walls
# ============================================================================
wall_nu = 0.2  # Poisson's ratio for concrete
wall_E = 27898 * MPa  # Value taken from approved structural calcs
wall_G = wall_E / (2*(1 + wall_nu))

wall_transf_tag = 1
ops.geomTransf('PDelta', wall_transf_tag, 0, 1, 0)

def get_wall_prop(section_prop, orient):

    wall_length = section_prop[0]
    wall_thick = section_prop[1]

    wall_area = wall_length * wall_thick

    # Compute moment of inertias and assign transformation tags based on wall orientation
    if orient == 'EW': # Wall orientation West-East
        wall_Iy = 0.5 * wall_length * (wall_thick**3) / 12
        wall_Iz = 0.5 *(wall_length**3) * wall_thick / 12

    else:  # Wall orientation is North-South
        wall_Iy = 0.5 * (wall_length**3) * wall_thick / 12
        wall_Iz = 0.5 * wall_length * (wall_thick**3) / 12

    return wall_area, wall_Iy, wall_Iz


def create_walls(floor_num):

    wall_tag = int('2' + floor_num + '01')  # 20101

    for ii in range(len(wall_prop)):
        wall_A, wall_Iy, wall_Iz = get_wall_prop(wall_prop[ii], wall_orient[ii])

        # element('ElasticTimoshenkoBeam', eleTag, *eleNodes, E_mod, G_mod, Area, Jxx, Iy, Iz, Avy, Avz, transfTag, <'-mass', massDens>, <'-cMass'>)
        ops.element('ElasticTimoshenkoBeam', wall_tag, lfre_node_tags.iloc[ii][floor_num] - 10000, lfre_node_tags.iloc[ii][floor_num], wall_E, wall_G, wall_A, 0.0, wall_Iy, wall_Iz, wall_A, wall_A, wall_transf_tag)

        # Update reference to wall_tag
        wall_tag += 1


# ============================================================================
# Create wall rigid links
# ============================================================================
bm_nu = 0.28  # Poisson's ratio for steel
bm_d = 602 * mm
bm_A = 13000 * mm**2
bm_E = 210 * GPa
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

ops.geomTransf('PDelta', wall_link_transf_tag_x, 0, -1, 0)
ops.geomTransf('PDelta', wall_link_transf_tag_y, 1, 0, 0)  # -1, 0, 0


def create_wall_rigid_links(floor_num):

    wall_rigid_tag = int('3' + floor_num + '01') # 30101

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
# Create colmns
# ============================================================================
col_nu = 0.28  # Poisson's ratio for steel
col_E = 210 * GPa

# The geometric properties of the columns oriented in the East-West direction will be defined using a W14x132 (metric W360x196)
col_A_EW = 25000 * mm**2
col_G_EW = col_E / (2*(1 + col_nu))
col_Iy_EW = 228 * 1E6 * mm**4   # weak Iyy
col_Iz_EW = 637 * 1E6 * mm**4   # strong Ixx
col_J_EW = 5120 * 1E3 * mm**4

col_transf_tag = 4
ops.geomTransf('PDelta', col_transf_tag, 0, 1, 0)  # Same Geometric transformation applies to columns in both directions

# The geometric properties of the columns oriented in the North-South direction will be defined using a W14x132 (metric W360x196)
col_A_NS = 25000 * mm**2
col_G_NS = col_E / (2*(1 + col_nu))
col_Iy_NS = 637 * 1E6 * mm**4   # strong Ixx
col_Iz_NS = 228 * 1E6 * mm**4   # weak Iyy
col_J_NS = 5120 * 1E3 * mm**4


def create_columns(floor_num):

    col_tag = int('2' + floor_num + '11')  # 20111

    ops.element('elasticBeamColumn', col_tag, lfre_node_tags.loc['col1'][floor_num] - 10000, lfre_node_tags.loc['col1'][floor_num], col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 1
    ops.element('elasticBeamColumn', col_tag + 1, lfre_node_tags.loc['col2'][floor_num] - 10000, lfre_node_tags.loc['col2'][floor_num], col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 2
    ops.element('elasticBeamColumn', col_tag + 2, lfre_node_tags.loc['col3'][floor_num] - 10000, lfre_node_tags.loc['col3'][floor_num], col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 3

    ops.element('elasticBeamColumn', col_tag + 3, lfre_node_tags.loc['col4'][floor_num] - 10000, lfre_node_tags.loc['col4'][floor_num], col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 4
    ops.element('elasticBeamColumn', col_tag + 4, lfre_node_tags.loc['col5'][floor_num] - 10000, lfre_node_tags.loc['col5'][floor_num], col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 5
    ops.element('elasticBeamColumn', col_tag + 5, lfre_node_tags.loc['col6'][floor_num] - 10000, lfre_node_tags.loc['col6'][floor_num], col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 6

    ops.element('elasticBeamColumn', col_tag + 6, lfre_node_tags.loc['col7'][floor_num] - 10000, lfre_node_tags.loc['col7'][floor_num], col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 7
    ops.element('elasticBeamColumn', col_tag + 7, lfre_node_tags.loc['col8'][floor_num] - 10000, lfre_node_tags.loc['col8'][floor_num], col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 8
    ops.element('elasticBeamColumn', col_tag + 8, lfre_node_tags.loc['col9'][floor_num] - 10000, lfre_node_tags.loc['col9'][floor_num], col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 9

    ops.element('elasticBeamColumn', col_tag + 9, lfre_node_tags.loc['col10'][floor_num] - 10000, lfre_node_tags.loc['col10'][floor_num], col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 10
    ops.element('elasticBeamColumn', col_tag + 10, lfre_node_tags.loc['col11'][floor_num] - 10000, lfre_node_tags.loc['col11'][floor_num], col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 11
    ops.element('elasticBeamColumn', col_tag + 11, lfre_node_tags.loc['col12'][floor_num] - 10000, lfre_node_tags.loc['col12'][floor_num], col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 12


# Create pvd recorder
record_direc = './pvd/'
os.makedirs(record_direc, exist_ok=True)
ops.recorder('PVD', record_direc, '-precision', 3, '-dT', 1, *['mass', 'reaction', 'eigen', 10])
# ops.recorder('PVD', record_direc, '-precision', 3, '-dT', 1, *['mass', 'reaction'])

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

# ops.recorder('Element', '-file', grav_direc + 'colForces.txt', '-ele', 20111, 20112, '-dof', 3, 'globalForce')
# ops.recorder('Node', '-file', grav_direc + 'nodeRxn.txt', '-node', *lfre_node_tags['00'].tolist(), '-dof', 3, 'reaction')


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
# Nonlinear static pushover
# ============================================================================
push = 0
push_direc = 1 # Direction for pushover; 1: x-axis; 2: y-xis

if push:
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
    ops.recorder('Node', '-file', record_direc + 'baseShearX.txt', '-node', *lfre_node_tags['00'].tolist(), '-dof', 1, 'reaction')
    ops.recorder('Node', '-file', record_direc + 'baseShearY.txt', '-node', *lfre_node_tags['00'].tolist(), '-dof', 2, 'reaction')
    # ops.recorder('Node', '-file', record_direc + 'baseShearX_Y.txt', '-node', *lfre_node_tags['00'].tolist(), '-dof', *[1, 2], 'reaction')

    # Recorder for roof COM displacement
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


# Clear model
ops.wipe()
