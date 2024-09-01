# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:57:06 2024

@author: udu
"""

# import os
import sys
import opensees.openseespy as ops
# import opsvis as opsv
import numpy as np
# import pandas as pd

import time
start = time.time()

# Append directory of helper functions to Python Path
sys.path.append('../')

from generate_rcsw_mvlem_model import build_model, run_eigen_gravity_analysis, run_elf_analysis, run_mrsa
from mvlem_helper_functions.rcsw_mrsa_demands import get_mvlem_element_demands

# ============================================================================
# CREATE MODEL
# ============================================================================
build_model('rigid')

# ============================================================================
# GRAVITY & EIGEN ANALYSIS
# ============================================================================
angular_freq, periods, modal_prop = run_eigen_gravity_analysis()

# # Load reaction forces due to gravity
grav_nodal_rxn = np.loadtxt('./gravity_results/' + 'nodeRxn.txt').T
grav_col_forces = np.loadtxt('./gravity_results/' + 'colForces.txt').T
grav_wall_nodal_forces = np.loadtxt('./gravity_results/' + 'WallForces.txt').T

# # Convert nodal rxns to dataframe
# grav_nodal_rxn_combined = pd.DataFrame(process_mvlem_element_demands(grav_nodal_rxn), columns=['Fz (kN)'],
#                                   index=list(wall_prop_df.index) + list(col_coords_df.index))

# # Extract axial forces from element demands.
# 'These values should equal `grav_nodal_wall_rxn`'
(_, _, grav_wall_elem_Fz, _, _, _) = get_mvlem_element_demands(grav_wall_nodal_forces)


# # ============================================================================
# # ELF ANALYSIS
# # ============================================================================
# elf_base_shear, elf_nodal_rxn, elf_col_demands, elf_wall_demands = run_elf_analysis(periods)

# # ============================================================================
# # MRSA
# # ============================================================================
# (mrsa_base_shearX, mrsa_base_shearY, elf_mrsaX_scale_factor, elf_mrsaY_scale_factor,
#      mrsaX_rcWall_Fx, mrsaX_rcWall_Fy,  mrsaX_rcWall_Fz, mrsaX_rcWall_Mx,  mrsaX_rcWall_My,  mrsaX_rcWall_Mz,
#      mrsaY_rcWall_Fx, mrsaY_rcWall_Fy,  mrsaY_rcWall_Fz, mrsaY_rcWall_Mx,  mrsaY_rcWall_My,  mrsaY_rcWall_Mz) = run_mrsa(angular_freq, elf_base_shear)

# Clear model
ops.wipe()

end = time.time()

print("Elapsed time = {:.2f} secs".format(end-start))




