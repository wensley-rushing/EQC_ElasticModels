# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:13:09 2024

@author: Uzo Uwaoma - udu@uw.edu
"""

def create_beam_recorders(ops, direc):

    ops.recorder('Element', '-file', direc + 'floor01_beamResp.txt', '-precision', 9, '-region', 201, 'force')
    ops.recorder('Element', '-file', direc + 'floor02_beamResp.txt', '-precision', 9, '-region', 202, 'force')
    ops.recorder('Element', '-file', direc + 'floor03_beamResp.txt', '-precision', 9, '-region', 203, 'force')
    ops.recorder('Element', '-file', direc + 'floor04_beamResp.txt', '-precision', 9, '-region', 204, 'force')
    ops.recorder('Element', '-file', direc + 'floor05_beamResp.txt', '-precision', 9, '-region', 205, 'force')
    ops.recorder('Element', '-file', direc + 'floor06_beamResp.txt', '-precision', 9, '-region', 206, 'force')
    ops.recorder('Element', '-file', direc + 'floor07_beamResp.txt', '-precision', 9, '-region', 207, 'force')
    ops.recorder('Element', '-file', direc + 'floor08_beamResp.txt', '-precision', 9, '-region', 208, 'force')
    ops.recorder('Element', '-file', direc + 'floor09_beamResp.txt', '-precision', 9, '-region', 209, 'force')
    ops.recorder('Element', '-file', direc + 'floor10_beamResp.txt', '-precision', 9, '-region', 210, 'force')
    ops.recorder('Element', '-file', direc + 'floor11_beamResp.txt', '-precision', 9, '-region', 211, 'force')


def create_column_recorders(ops, direc):
    ops.recorder('Element', '-file', direc + 'floor01_colResp.txt', '-precision', 9, '-region', 301, 'force')
    ops.recorder('Element', '-file', direc + 'floor02_colResp.txt', '-precision', 9, '-region', 302, 'force')
    ops.recorder('Element', '-file', direc + 'floor03_colResp.txt', '-precision', 9, '-region', 303, 'force')
    ops.recorder('Element', '-file', direc + 'floor04_colResp.txt', '-precision', 9, '-region', 304, 'force')
    ops.recorder('Element', '-file', direc + 'floor05_colResp.txt', '-precision', 9, '-region', 305, 'force')
    ops.recorder('Element', '-file', direc + 'floor06_colResp.txt', '-precision', 9, '-region', 306, 'force')
    ops.recorder('Element', '-file', direc + 'floor07_colResp.txt', '-precision', 9, '-region', 307, 'force')
    ops.recorder('Element', '-file', direc + 'floor08_colResp.txt', '-precision', 9, '-region', 308, 'force')
    ops.recorder('Element', '-file', direc + 'floor09_colResp.txt', '-precision', 9, '-region', 309, 'force')
    ops.recorder('Element', '-file', direc + 'floor10_colResp.txt', '-precision', 9, '-region', 310, 'force')
    ops.recorder('Element', '-file', direc + 'floor11_colResp.txt', '-precision', 9, '-region', 311, 'force')


def create_wall_recorders(ops, direc):
    ops.recorder('Element', '-file', direc + 'floor01_wallResp.txt', '-precision', 9, '-region', 401, 'force')
    ops.recorder('Element', '-file', direc + 'floor02_wallResp.txt', '-precision', 9, '-region', 402, 'force')
    ops.recorder('Element', '-file', direc + 'floor03_wallResp.txt', '-precision', 9, '-region', 403, 'force')
    ops.recorder('Element', '-file', direc + 'floor04_wallResp.txt', '-precision', 9, '-region', 404, 'force')
    ops.recorder('Element', '-file', direc + 'floor05_wallResp.txt', '-precision', 9, '-region', 405, 'force')
    ops.recorder('Element', '-file', direc + 'floor06_wallResp.txt', '-precision', 9, '-region', 406, 'force')
    ops.recorder('Element', '-file', direc + 'floor07_wallResp.txt', '-precision', 9, '-region', 407, 'force')
    ops.recorder('Element', '-file', direc + 'floor08_wallResp.txt', '-precision', 9, '-region', 408, 'force')
    ops.recorder('Element', '-file', direc + 'floor09_wallResp.txt', '-precision', 9, '-region', 409, 'force')
    ops.recorder('Element', '-file', direc + 'floor10_wallResp.txt', '-precision', 9, '-region', 410, 'force')
    ops.recorder('Element', '-file', direc + 'floor11_wallResp.txt', '-precision', 9, '-region', 411, 'force')


def create_wall_rigid_link_recorders(ops, direc):
    ops.recorder('Element', '-file', direc + 'floor01_wallRigidLinkResp.txt', '-precision', 9, '-region', 501, 'force')
    ops.recorder('Element', '-file', direc + 'floor02_wallRigidLinkResp.txt', '-precision', 9, '-region', 502, 'force')
    ops.recorder('Element', '-file', direc + 'floor03_wallRigidLinkResp.txt', '-precision', 9, '-region', 503, 'force')
    ops.recorder('Element', '-file', direc + 'floor04_wallRigidLinkResp.txt', '-precision', 9, '-region', 504, 'force')
    ops.recorder('Element', '-file', direc + 'floor05_wallRigidLinkResp.txt', '-precision', 9, '-region', 505, 'force')
    ops.recorder('Element', '-file', direc + 'floor06_wallRigidLinkResp.txt', '-precision', 9, '-region', 506, 'force')
    ops.recorder('Element', '-file', direc + 'floor07_wallRigidLinkResp.txt', '-precision', 9, '-region', 507, 'force')
    ops.recorder('Element', '-file', direc + 'floor08_wallRigidLinkResp.txt', '-precision', 9, '-region', 508, 'force')
    ops.recorder('Element', '-file', direc + 'floor09_wallRigidLinkResp.txt', '-precision', 9, '-region', 509, 'force')
    ops.recorder('Element', '-file', direc + 'floor10_wallRigidLinkResp.txt', '-precision', 9, '-region', 510, 'force')
    ops.recorder('Element', '-file', direc + 'floor11_wallRigidLinkResp.txt', '-precision', 9, '-region', 511, 'force')

    ops.recorder('Element', '-file', direc + 'floor01_wallRigid2elem.txt', '-precision', 9, '-ele', 50101, 50102, 'force')
