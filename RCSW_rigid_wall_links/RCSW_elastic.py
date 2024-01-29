# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:43:33 2023

@author: Uzo Uwaoma - udu@uw.edu
"""
# Import required modules
import os
import sys
import openseespy.opensees as ops
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
from helper_functions.rcsw_wall_rigid_links import create_wall_rigid_links
from helper_functions.rcsw_create_regions import create_column_and_wall_regions

from helper_functions.eigen_analysis import run_eigen_analysis
from helper_functions.cqc_modal_combo import modal_combo
from helper_functions.run_mrsa import perform_rcsw_mrsa
from helper_functions.rcsw_mrsa_demands import get_mrsa_wall_demands, get_mrsa_wall_rigid_links_demands

from helper_functions.get_beam_col_demands import process_beam_col_resp
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
conc_mass_rho = conc_rho / grav_metric
conc_fcp = 40 * MPa  # Compressive strength of concrete
conc_E = 27898 * MPa  # Value taken from approved structural calcs
conc_nu = 0.2  # Poisson's ratio of concrete
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
               'col8': [21.7, 16.625],
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
    mesh_size = 1.5 * m  # Mesh size
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
shell_sect_tag = 1

slab_thick = 165 * mm

slab_in_plane_modif = 10  # 1E3
slab_out_plane_modif = 0.05
shell_E =  slab_in_plane_modif * conc_E # Modulus of concrete
shell_nu = conc_nu  # Poisson's ratio

# ============================================================================
# Initialize dictionary to store node tags of COM for all floors
# Initialize dictionary to store total mass of each floor
# ============================================================================
com_node_tags = {}
total_floor_mass = {}

ductility_factor = 1.25  # RCSW

# ============================================================================
# Define function to create a floor
# ============================================================================

def create_floor(elev, floor_num, floor_label=''):

    node_compile = []  # Store node tags grouped according to their y-coordinates

    # Create nodes
    node_num = int(floor_num + '1000000')

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

        # Constraints for Rigid Diaphragm Primary node
        ops.fix(com_node, 0, 0, 1, 1, 1, 0)  # dx, dy, dz, rx, ry, rz

        # Impose rigid diaphragm constraint
        ops.rigidDiaphragm(3, com_node, *floor_node_tags)

        # Add COM node tag of current floor to dictionary
        com_node_tags[floor_num] = com_node

        # Create columns & walls
        create_columns(floor_num)
        create_walls(floor_num)
        create_wall_rigid_links(ops, floor_num, wall_ends_node_tags, lfre_node_tags, wall_link_prop)

    print('Floor ' + floor_num + ' created')


# ============================================================================
# Create material properties for wall
# ============================================================================
wall_nu = conc_nu # Poisson's ratio for concrete
wall_E = conc_E
wall_G = wall_E / (2*(1 + wall_nu))
wall_transf_tag = 1  # Walls in EW & NS direction have the same transformation


def get_wall_prop(section_prop, orient):

    wall_length = section_prop[0]
    wall_thick = section_prop[1]

    wall_area = wall_length * wall_thick

    # Compute moment of inertias (using reduced sectional properties) based on wall orientation
    if orient == 'EW': # Wall orientation West-East
        wall_Iy = 0.5 * wall_length * (wall_thick**3) / 12 * 0.00001
        wall_Iz = 0.5 *(wall_length**3) * wall_thick / 12

    else:  # Wall orientation is North-South
        wall_Iy = 0.5 * (wall_length**3) * wall_thick / 12
        wall_Iz = 0.5 * wall_length * (wall_thick**3) / 12 * 0.00001

    return wall_area, wall_Iy, wall_Iz


def create_walls(floor_num):

    wall_tag = int('4' + floor_num + '01')  # 40101

    for ii in range(len(wall_prop)):
        wall_A, wall_Iy, wall_Iz = get_wall_prop(wall_prop[ii], wall_orient[ii])

        # element('ElasticTimoshenkoBeam', eleTag, *eleNodes, E_mod, G_mod, Area, Jxx, Iy, Iz, Avy, Avz, transfTag, <'-mass', massDens>, <'-cMass'>)
        ops.element('ElasticTimoshenkoBeam', wall_tag, lfre_node_tags.iloc[ii][floor_num] - 10000000,
                    lfre_node_tags.iloc[ii][floor_num], wall_E, wall_G, wall_A, 0.0, wall_Iy, wall_Iz, wall_A, wall_A, wall_transf_tag)

        # Update reference to wall_tag
        wall_tag += 1


# ============================================================================
# Define material/geometric properties for wall rigid links
# ============================================================================
bm_nu = 0.28  # Poisson's ratio for steel
bm_d = 602 * mm  # Using properties of W24x68 (metric W610x101)
bm_A = 13000 * mm**2
bm_E = steelE
bm_G = bm_E / (2*(1 + bm_nu))
bm_I = 762 * 1E6 * mm**4   # strong axis
bm_J = 778 * 1E3 * mm**4

wall_link_A = bm_A
wall_link_E = bm_E * 1E3
wall_link_G = bm_G
wall_link_J = bm_J
wall_link_I = bm_I

wall_link_transf_tag_x = 2 # Walls oriented in Global-X direction
wall_link_transf_tag_y = 3 # Walls oriented in Global-Y direction

wall_link_prop = [wall_link_A, wall_link_E, wall_link_G, wall_link_J,
                  wall_link_I, wall_link_transf_tag_x, wall_link_transf_tag_y]

# ============================================================================
# Create columns
# ============================================================================
# Only axial stiffness of gravity columns will be captured.
col_nu = 0.28  # Poisson's ratio for steel
col_E = steelE

# The geometric properties of the columns oriented in the East-West direction will be defined using a W14x132 (metric W360x196)
col_A_EW = 25000 * mm**2
col_G_EW = col_E / (2*(1 + col_nu))
col_stiff_modif = 1e-6
col_Iy_EW = col_stiff_modif * 228 * 1E6 * mm**4   # weak Iyy
col_Iz_EW = col_stiff_modif * 637 * 1E6 * mm**4   # strong Ixx
col_J_EW = col_stiff_modif * 5120 * 1E3 * mm**4

col_transf_tag = 4

# The geometric properties of the columns oriented in the North-South direction will be defined using a W14x132 (metric W360x196)
col_A_NS = 25000 * mm**2
col_G_NS = col_E / (2*(1 + col_nu))
col_Iy_NS = col_stiff_modif * 637 * 1E6 * mm**4   # strong Ixx
col_Iz_NS = col_stiff_modif * 228 * 1E6 * mm**4   # weak Iyy
col_J_NS = col_stiff_modif * 5120 * 1E3 * mm**4


def create_columns(floor_num):

    col_tag = int('3' + floor_num + '01')  # 30101

    ops.element('elasticBeamColumn', col_tag, lfre_node_tags.loc['col1'][floor_num] - 10000000, lfre_node_tags.loc['col1'][floor_num],
                col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 1

    ops.element('elasticBeamColumn', col_tag + 1, lfre_node_tags.loc['col2'][floor_num] - 10000000, lfre_node_tags.loc['col2'][floor_num],
                col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 2

    ops.element('elasticBeamColumn', col_tag + 2, lfre_node_tags.loc['col3'][floor_num] - 10000000, lfre_node_tags.loc['col3'][floor_num],
                col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 3

    ops.element('elasticBeamColumn', col_tag + 3, lfre_node_tags.loc['col4'][floor_num] - 10000000, lfre_node_tags.loc['col4'][floor_num],
                col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 4

    ops.element('elasticBeamColumn', col_tag + 4, lfre_node_tags.loc['col5'][floor_num] - 10000000, lfre_node_tags.loc['col5'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 5

    ops.element('elasticBeamColumn', col_tag + 5, lfre_node_tags.loc['col6'][floor_num] - 10000000, lfre_node_tags.loc['col6'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 6

    ops.element('elasticBeamColumn', col_tag + 6, lfre_node_tags.loc['col7'][floor_num] - 10000000, lfre_node_tags.loc['col7'][floor_num],
                col_A_NS, col_E, col_G_NS, col_J_NS, col_Iy_NS, col_Iz_NS, col_transf_tag)  # Col 7

    ops.element('elasticBeamColumn', col_tag + 7, lfre_node_tags.loc['col8'][floor_num] - 10000000, lfre_node_tags.loc['col8'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 8

    ops.element('elasticBeamColumn', col_tag + 8, lfre_node_tags.loc['col9'][floor_num] - 10000000, lfre_node_tags.loc['col9'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 9

    ops.element('elasticBeamColumn', col_tag + 9, lfre_node_tags.loc['col10'][floor_num] - 10000000, lfre_node_tags.loc['col10'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 10

    ops.element('elasticBeamColumn', col_tag + 10, lfre_node_tags.loc['col11'][floor_num] - 10000000, lfre_node_tags.loc['col11'][floor_num],
                col_A_EW, col_E, col_G_EW, col_J_EW, col_Iy_EW, col_Iz_EW, col_transf_tag)  # Col 11

    ops.element('elasticBeamColumn', col_tag + 11, lfre_node_tags.loc['col12'][floor_num] - 10000000, lfre_node_tags.loc['col12'][floor_num],
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
    # https://openseespydoc.readthedocs.io/en/latest/src/elasticMembranePlateSection.html
    ops.section('ElasticMembranePlateSection', shell_sect_tag, shell_E, shell_nu, slab_thick, 0.0, slab_out_plane_modif)

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
    # Create regions for steel columns & RC Walls & rigid wall links based on floor
    # ============================================================================
    # Get all element tags
    elem_tags = ops.getEleTags()

    floor_nums = ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11']

    col_tags = []
    wall_tags = []
    wall_rigid_link_tags = []

    for floor in floor_nums:
        floor_col_tags = []
        floor_wall_tags = []
        floor_wall_rigid_link_tags = []

        for tag in elem_tags:

            # Only select column elements
            if str(tag).startswith('3' + floor):
                floor_col_tags.append(tag)

            # Only select wall elements
            elif str(tag).startswith('4' + floor):
                floor_wall_tags.append(tag)

            # Only select wall rigid link elements
            elif str(tag).startswith('5' + floor):
                floor_wall_rigid_link_tags.append(tag)

        col_tags.append(floor_col_tags)
        wall_tags.append(floor_wall_tags)
        wall_rigid_link_tags.append(floor_wall_rigid_link_tags)

    # Create regions for columns, walls, and rigid wall links for use in Recorders
    create_column_and_wall_regions(ops, col_tags, wall_tags, wall_rigid_link_tags)


# Generate model
build_model()

# Create pvd recorder
record_direc = './pvd/'
os.makedirs(record_direc, exist_ok=True)
ops.recorder('PVD', record_direc, '-precision', 3, '-dT', 1, *['disp', 'reaction', 'mass', 'eigen', 10])

# ============================================================================
# Eigen Analysis
# ============================================================================
num_modes = 10
damping_ratio = 0.05

angular_freq, periods, modal_prop = run_eigen_analysis(ops, num_modes, damping_ratio, './', 'RCSW')

# ============================================================================
# Gravity analysis
# ============================================================================
# Create recorder
grav_direc = './gravity_results/'
os.makedirs(grav_direc, exist_ok=True)
ops.recorder('Element', '-file', grav_direc + 'colForces.txt', '-region', 301, '-dof', 1,2, 3, 4, 5, 6, 'globalForce')  # 1st floor columns
ops.recorder('Element', '-file', grav_direc + 'WallForces.txt', '-region', 401, '-dof', 1,2, 3, 4, 5, 6, 'globalForce')  # 1st floor walls
ops.recorder('Node', '-file', grav_direc + 'nodeRxn.txt', '-node', *lfre_node_tags['00'].tolist(), '-dof', 3, 'reaction')

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

# Load reaction forces due to gravity
grav_rxn_forces = np.loadtxt(grav_direc + 'nodeRxn.txt').T

# ============================================================================
# Modal Response Spectrum Analysis
# ============================================================================

# Load spectral accelerations and periods for response spectrum
# Using Wellington's Hazard
spect_acc = np.loadtxt('../nz_spectral_acc.txt') / ductility_factor
spect_periods = np.loadtxt('../nz_periods.txt')


perform_rcsw_mrsa(ops, spect_acc, spect_periods, num_modes, './mrsa_results/dir',
             wall_ends_node_tags, lfre_node_tags, com_node_tags)


# Clear model
ops.wipe()

print('\nMRSA completed.')
print('======================================================')

# ============================================================================
# Post-process MRSA results
# ============================================================================
mrsa_base_shearX = modal_combo(np.loadtxt('./mrsa_results/dirX/baseShearX.txt'), angular_freq, damping_ratio, num_modes).sum()
mrsa_base_shearY = modal_combo(np.loadtxt('./mrsa_results/dirY/baseShearY.txt'), angular_freq, damping_ratio, num_modes).sum()

# Get MRSA demands in walls
[mrsaX_rcWall_Fx, mrsaX_rcWall_Fy,  mrsaX_rcWall_Fz,
  mrsaX_rcWall_Mx,  mrsaX_rcWall_My,  mrsaX_rcWall_Mz] =  get_mrsa_wall_demands(modal_combo, './mrsa_results/dirX/', angular_freq,
                                                                          damping_ratio, num_modes)

[mrsaY_rcWall_Fx, mrsaY_rcWall_Fy,  mrsaY_rcWall_Fz,
  mrsaY_rcWall_Mx,  mrsaY_rcWall_My,  mrsaY_rcWall_Mz] =  get_mrsa_wall_demands(modal_combo, './mrsa_results/dirY/', angular_freq,
                                                                          damping_ratio, num_modes)

# Get MRSA demands in wall rigid links
[mrsaX_wall_links_Fx, mrsaX_wall_links_Fy,
  mrsaX_wall_links_Fz, mrsaX_wall_links_Mx,
  mrsaX_wall_links_My,  mrsaX_wall_links_Mz] = get_mrsa_wall_rigid_links_demands(modal_combo, './mrsa_results/dirX/', angular_freq,
                                                                                 damping_ratio, num_modes)

[mrsaY_wall_links_Fx, mrsaY_wall_links_Fy,
  mrsaY_wall_links_Fz, mrsaY_wall_links_Mx,
  mrsaY_wall_links_My,  mrsaY_wall_links_Mz] = get_mrsa_wall_rigid_links_demands(modal_combo, './mrsa_results/dirX/', angular_freq,
                                                                                 damping_ratio, num_modes)


# """
# ============================================================================
# Compute Torsional Irregularity Ratio (TIR)
# ============================================================================
# Obtain peak total response for corner node displacments

# ===== MRSA - X
lower_left_corner_dispX = modal_combo(np.loadtxt('./mrsa_results/dirX/lowerLeftCornerDisp.txt'), angular_freq, damping_ratio, num_modes)
upper_right_corner_dispX = modal_combo(np.loadtxt('./mrsa_results/dirX/upperRightCornerDisp.txt'), angular_freq, damping_ratio, num_modes)
lower_right_corner_dispX = modal_combo(np.loadtxt('./mrsa_results/dirX/lowerRightCornerDisp.txt'), angular_freq, damping_ratio, num_modes)

tir_x_edgeE = np.maximum(upper_right_corner_dispX, lower_right_corner_dispX) / (0.5*(upper_right_corner_dispX + lower_right_corner_dispX))  # Right edge of building plan
tir_x_edgeF = np.maximum(lower_left_corner_dispX, lower_right_corner_dispX) / (0.5*(lower_left_corner_dispX + lower_right_corner_dispX))    # Bottom edge of building plan

# ===== MRSA - Y
lower_left_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/lowerLeftCornerDisp.txt'), angular_freq, damping_ratio, num_modes)
upper_right_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/upperRightCornerDisp.txt'), angular_freq, damping_ratio, num_modes)
lower_right_corner_dispY = modal_combo(np.loadtxt('./mrsa_results/dirY/lowerRightCornerDisp.txt'), angular_freq, damping_ratio, num_modes)

tir_y_edgeE = np.maximum(upper_right_corner_dispY, lower_right_corner_dispY) / (0.5*(upper_right_corner_dispY + lower_right_corner_dispY))  # Right edge of building plan
tir_y_edgeF = np.maximum(lower_left_corner_dispY, lower_right_corner_dispY) / (0.5*(lower_left_corner_dispY + lower_right_corner_dispY))    # Bottom edge of building plan
# """

# ============================================================================
# Perform ELF
# ============================================================================
spectral_shape_factor = spectral_shape_fac(periods[0])
hazard_factor = 0.13
return_per_factor_sls = 0.25
return_per_factor_uls = 1.3
fault_factor = 1.0
perform_factor = 0.7
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


# """
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

# P-Delta Method B: (NZS 1170.5:2004 - Sect. 6.5.4.2 & Commentary Sect. C6.5.4.2)
# Modal combination on peak COM displacements from MRSA
mrsa_total_com_dispX = modal_combo(mrsa_com_dispX, angular_freq, damping_ratio, num_modes)
mrsa_total_com_dispY = modal_combo(mrsa_com_dispY, angular_freq, damping_ratio, num_modes)

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
        print('\nNow commencing static analysis using torsional moments for '
              + torsional_folder[jj] + ' ' + torsional_direc[ii] + ' direction.')

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

            if torsional_direc[ii] == 'X': # Torsional moment about z-axis & PDelta "Method B" forces are applied
                ops.load(com_nodes[kk], lateral_forces_pDeltaX[kk], 0., 0., 0., 0., torsional_mom_x[kk] * torsional_sign[jj])
                # print(lateral_forces_pDeltaX[kk], torsional_mom_x[kk])

            elif torsional_direc[ii] == 'Y':  # Torsional moment about z-axis & PDelta "Method B" forces are applied
                ops.load(com_nodes[kk], 0., lateral_forces_pDeltaY[kk], 0., 0., 0., torsional_mom_y[kk] * torsional_sign[jj])
                # print(lateral_forces_pDeltaX[kk], torsional_mom_x[kk])

        # Create directory to save results
        accident_torsion_res_folder = './accidental_torsion_results/' + torsional_folder[jj] + torsional_direc[ii] + '/'
        os.makedirs(accident_torsion_res_folder, exist_ok=True)

        # Create recorders for column response in direction of excitation
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor01_colResp.txt',
                      '-precision', 9, '-region', 301, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor02_colResp.txt',
                      '-precision', 9, '-region', 302, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor03_colResp.txt',
                      '-precision', 9, '-region', 303, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor04_colResp.txt',
                      '-precision', 9, '-region', 304, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor05_colResp.txt',
                      '-precision', 9, '-region', 305, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor06_colResp.txt',
                      '-precision', 9, '-region', 306, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor07_colResp.txt',
                      '-precision', 9, '-region', 307, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor08_colResp.txt',
                      '-precision', 9, '-region', 308, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor09_colResp.txt',
                      '-precision', 9, '-region', 309, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor10_colResp.txt',
                      '-precision', 9, '-region', 310, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor11_colResp.txt',
                      '-precision', 9, '-region', 311, 'force')

        # Create recorders for wall response in direction of excitation
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor01_wallResp.txt',
                      '-precision', 9, '-region', 401, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor02_wallResp.txt',
                      '-precision', 9, '-region', 402, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor03_wallResp.txt',
                      '-precision', 9, '-region', 403, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor04_wallResp.txt',
                      '-precision', 9, '-region', 404, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor05_wallResp.txt',
                      '-precision', 9, '-region', 405, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor06_wallResp.txt',
                      '-precision', 9, '-region', 406, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor07_wallResp.txt',
                      '-precision', 9, '-region', 407, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor08_wallResp.txt',
                      '-precision', 9, '-region', 408, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor09_wallResp.txt',
                      '-precision', 9, '-region', 409, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor10_wallResp.txt',
                      '-precision', 9, '-region', 410, 'force')
        ops.recorder('Element', '-file', accident_torsion_res_folder + 'floor11_wallResp.txt',
                      '-precision', 9, '-region', 411, 'force')

        # Recorders for COM displacement
        ops.recorder('Node', '-file', accident_torsion_res_folder + 'COM_disp' + torsional_direc[ii] + '.txt',
                      '-node', *list(com_node_tags.values()), '-dof', elf_dof[ii], 'disp')

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
np.savetxt('driftX.txt', story_driftX, fmt='%.2f')
np.savetxt('driftY.txt', story_driftY, fmt='%.2f')

# ============================================================================
# Post-process MRSA & accidental torsion results
# ============================================================================
col_demands_X = process_beam_col_resp('col', './mrsa_results/dirX/', './accidental_torsion_results/positiveX/',
                                      './accidental_torsion_results/negativeX/', angular_freq, damping_ratio,
                                      num_modes, elf_mrsaX_scale_factor, pdelta_fac)

col_demands_Y = process_beam_col_resp('col', './mrsa_results/dirY/', './accidental_torsion_results/positiveY/',
                                      './accidental_torsion_results/negativeY/', angular_freq, damping_ratio,
                                      num_modes, elf_mrsaY_scale_factor, pdelta_fac)

wall_demands_X, wall_axialLoad_X, wall_mom_X = process_beam_col_resp('wall', './mrsa_results/dirX/', './accidental_torsion_results/positiveX/',
                                      './accidental_torsion_results/negativeX/', angular_freq, damping_ratio,
                                      num_modes, elf_mrsaX_scale_factor, pdelta_fac)

wall_demands_Y, wall_axialLoad_Y, wall_mom_Y = process_beam_col_resp('wall', './mrsa_results/dirY/', './accidental_torsion_results/positiveY/',
                                      './accidental_torsion_results/negativeY/', angular_freq, damping_ratio,
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

# # Generate story drift plots
# fig, ax = plt.subplots(1, 2, figsize=(6.0, 7.5), sharex=True, sharey=True, constrained_layout=True)
# fig.suptitle('Story drift ratios', fontdict=title_font)

# ax[0].vlines(story_driftX[0], 0.0, elev[0])
# ax[1].vlines(story_driftY[0], 0.0, elev[0])

# for ii in range(1, len(story_driftX)):
#     ax[0].hlines(elev[ii-1], story_driftX[ii-1], story_driftX[ii])
#     ax[0].vlines(story_driftX[ii],  elev[ii-1], elev[ii])

#     ax[1].hlines(elev[ii-1], story_driftY[ii-1], story_driftY[ii])  # Correct
#     ax[1].vlines(story_driftY[ii],  elev[ii-1], elev[ii])


# ax[0].set_title('X - Direction', fontsize=12, family='Times New Roman')
# ax[1].set_title('Y- Direction', fontsize=12, family='Times New Roman')

# ax[0].set_ylabel('Story elevation (m)', fontdict=axes_font)

# for axx in ax.flat:
#     axx.set_xlim(0.0)
#     axx.set_ylim(0.0, elev[-1])

#     axx.grid(True, which='major', axis='both', ls='-.', linewidth=0.6)

#     axx.set_yticks(elev)

#     axx.set_xlabel('Story drift ratio (%)', fontdict=axes_font)

#     axx.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
#     axx.tick_params(axis='both', direction='in', colors='grey', labelcolor='grey', zorder=3.0, labelsize=8.0)

# # plt.savefig('DriftPlots.png', dpi=1200)


# Compute wall reinforcing ratios
steel_fy = 500 * MPa
wall_reinf_ratio = pd.DataFrame(index=wall_mom_X.index, columns=wall_mom_X.columns)

wall_index = 0

for col in wall_mom_X:

    num_floors = wall_mom_X.shape[0]
    reinf_ratio = np.zeros(num_floors)

    wall_mom = wall_mom_X[col].to_numpy()
    wall_load = -1.0 * wall_axialLoad_X[col].to_numpy()

    for ii in range(num_floors):
        reinf_ratio[ii] = get_wall_reinf_ratio(wall_prop[wall_index][0], wall_prop[wall_index][1],
                              wall_load[ii], wall_mom[ii], conc_fcp, steel_fy, steelE)

    wall_reinf_ratio[col] = reinf_ratio
    wall_index += 1

wall_Pdel_posX = np.loadtxt('./accidental_torsion_results/positiveX/floor01_wallResp.txt')
wall_Pdel_posY = np.loadtxt('./accidental_torsion_results/positiveY/floor01_wallResp.txt')
wall_Pdel_negY = np.loadtxt('./accidental_torsion_results/negativeY/floor01_wallResp.txt')
wall_Pdel_negX = np.loadtxt('./accidental_torsion_results/negativeX/floor01_wallResp.txt')

# """
