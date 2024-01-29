# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:05:00 2023

@author: Uzo Uwaoma - udu@uw.edu
"""


def create_column_and_wall_regions(ops, col_tags, wall_tags, wall_rigid_link_tags=None):
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

    if wall_rigid_link_tags:
        # Wall rigid links for elasticBeamColumn RCSW models
        ops.region(501, '-eleOnly', *wall_rigid_link_tags[0])  # Region for all wall rigid links on 1st floor
        ops.region(502, '-eleOnly', *wall_rigid_link_tags[1])  # Region for all wall rigid links on 2nd floor
        ops.region(503, '-eleOnly', *wall_rigid_link_tags[2])  # Region for all wall rigid links on 3rd floor
        ops.region(504, '-eleOnly', *wall_rigid_link_tags[3])  # Region for all wall rigid links on 4th floor
        ops.region(505, '-eleOnly', *wall_rigid_link_tags[4])  # Region for all wall rigid links on 5th floor
        ops.region(506, '-eleOnly', *wall_rigid_link_tags[5])  # Region for all wall rigid links on 6th floor
        ops.region(507, '-eleOnly', *wall_rigid_link_tags[6])  # Region for all wall rigid links on 7th floor
        ops.region(508, '-eleOnly', *wall_rigid_link_tags[7])  # Region for all wall rigid links on 8th floor
        ops.region(509, '-eleOnly', *wall_rigid_link_tags[8])  # Region for all wall rigid links on 9th floor
        ops.region(510, '-eleOnly', *wall_rigid_link_tags[9])  # Region for all wall rigid links on 10th floor
        ops.region(511, '-eleOnly', *wall_rigid_link_tags[10])  # Region for all wall rigid links on 11th floor


def create_beam_region(ops, beam_tags):
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

def create_column_region(ops, col_tags):
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

