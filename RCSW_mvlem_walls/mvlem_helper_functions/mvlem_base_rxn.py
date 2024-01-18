# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:11:53 2024

@author: Uzo Uwaoma - udu@uw.edu
"""

import numpy as np

def get_mvlem_base_rxn(nodal_reaction):
    """
    Helper function to extract & combine nodal reactions at the base.


    Each MVLEM wall has two nodes at its base.

    `nodal_reaction` parameter contains the nodal rxns at each of these nodes
    together with the rxns at the column bases.

    There are 10 walls, hence 20 nodes. Therefore the 1st 20
    values in `nodal_reaction` are the nodal rxns of the walls - this is
    enforced by how the recorder is set up.

    The rxn for each wall is the sum of the rxns at the 2 nodes forming
    its base.

    Therefore, extract the sum of base reactions for each wall
    then create a new array of base reactions for walls & columns.

    Parameters
    ----------
    nodal_reaction : numpy.array
        Array of nodal reactionss.

    """


    wall_rxn = nodal_reaction[0:20:2] + nodal_reaction[1:20:2]
    combined_nodal_rxn = np.hstack((wall_rxn, nodal_reaction[20:]))

    return combined_nodal_rxn
