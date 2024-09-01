# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:45:42 2023

@author: udu
"""
import numpy as np
import opensees.openseespy as ops


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


# Define tributary heights of cladding for each floor
clad_trib_height = {'01': 3.8,
                    '02': 3.1,
                    '03': 3.1,
                    '04': 3.1,
                    '05': 3.1,
                    '06': 3.1,
                    '07': 3.1,
                    '08': 3.1,
                    '09': 3.1,
                    '10': 3.1,
                    '11': 1.55}

def refine_mesh(coordinates, mesh_size):

    # Compute differences between coordinate points
    coord_diff = np.diff(coordinates)

    # Initialize list to store location of refined coordinates
    refined_coords = [coordinates[0]]

    for ii in range(len(coord_diff)):
        if coord_diff[ii] > mesh_size:
            refined_coords.append(list(np.arange(coordinates[ii], coordinates[ii+1], mesh_size)))

        refined_coords.append(coordinates[ii+1])

    # Compile refined coordinates into a single list
    combined_coords = []

    for item in refined_coords:
        if type(item) == list:
            for item_ in item:
                combined_coords.append(item_)
        else:
            combined_coords.append(item)

    # Remove duplicates
    combined_coords = sorted(list(set(combined_coords)))

    return combined_coords


def create_shell(floor_num, node_compile, shell_sect_tag, num_y_groups):
    shell_tag = int(floor_num + '20000')

    # Create timeSeries and Pattern for assigning load
    ts_tag = int(floor_num)
    pattern_tag = int(floor_num)

    ops.timeSeries('Constant', ts_tag)
    ops.pattern('Plain', pattern_tag, ts_tag)

    for jj in range(num_y_groups-1):

        node_down = node_compile[jj]
        node_up = node_compile[jj+1]

        # Account for difference in length of nodes just below and above rentrant boundary line
        if len(node_down) != len(node_up):
            node_down = node_down[-len(node_up):]

        for kk in range(len(node_down) - 1):
            lwr_left_node = node_down[kk]
            lwr_right_node = node_down[kk + 1]
            upr_right_node = node_up[kk + 1]
            upr_left_node = node_up[kk]

            # Calculate area of shell
            shell_area = (ops.nodeCoord(lwr_right_node, 1) -  ops.nodeCoord(lwr_left_node, 1)) * (ops.nodeCoord(upr_left_node, 2) - ops.nodeCoord(lwr_left_node, 2))


            if floor_num != '11': # Factored area load all floors except the roof.
                area_load = 6.16 * kN / m **2

            else:  # Factored area load at roof
                area_load = 0.5 * kN / m **2

            shell_nodal_load = area_load * shell_area / 4  # in kN
            shell_nodal_mass = shell_nodal_load / grav_metric  # in kN-sec^2/m


            ops.element('ShellNLDKGQ', shell_tag, lwr_left_node, lwr_right_node, upr_right_node, upr_left_node, shell_sect_tag)

            # Assign load
            ops.load(lwr_left_node, 0, 0, -shell_nodal_load, 0, 0, 0)
            ops.load(lwr_right_node, 0, 0, -shell_nodal_load, 0, 0, 0)
            ops.load(upr_right_node, 0, 0, -shell_nodal_load, 0, 0, 0)
            ops.load(upr_left_node, 0, 0, -shell_nodal_load, 0, 0, 0)

            lwr_left_node_mass = ops.nodeMass(lwr_left_node, 1) + shell_nodal_mass
            lwr_right_node_mass = ops.nodeMass(lwr_right_node, 1) + shell_nodal_mass
            upr_right_node_mass = ops.nodeMass(upr_right_node, 1) + shell_nodal_mass
            upr_left_node_mass = ops.nodeMass(upr_left_node, 1) + shell_nodal_mass

            # Assign mass
            neglig_mass = 1E-8 * kN * sec**2 / m
            ops.mass(lwr_left_node, lwr_left_node_mass, lwr_left_node_mass, neglig_mass, neglig_mass, neglig_mass, neglig_mass)
            ops.mass(lwr_right_node, lwr_right_node_mass, lwr_right_node_mass, neglig_mass, neglig_mass, neglig_mass, neglig_mass)
            ops.mass(upr_right_node, upr_right_node_mass, upr_right_node_mass, neglig_mass, neglig_mass, neglig_mass, neglig_mass)
            ops.mass(upr_left_node, upr_left_node_mass, upr_left_node_mass, neglig_mass, neglig_mass, neglig_mass, neglig_mass)

            shell_tag += 1

        # print(lwr_right_node, lwr_left_node, upr_left_node, lwr_left_node)
        # print(ops.nodeCoord(lwr_right_node, 1), ops.nodeCoord(lwr_left_node, 1), ops.nodeCoord(upr_left_node, 2), ops.nodeCoord(lwr_left_node, 2))

        # Assign cladding loads to perimeter nodes
