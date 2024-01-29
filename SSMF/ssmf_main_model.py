# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:15:11 2024

@author: Uzo UWaoma - udu@uw.edu
"""

import sys
import openseespy.opensees as ops
import numpy as np
# import opsvis as opsv


# Append directory of helper functions to Python Path
sys.path.append('./')

from generate_ssmf_model import (build_model, run_eigen_gravity_analysis,
                                 run_elf_analysis, run_mrsa, run_pdelta_analysis,
                                 get_mrsa_and_torsional_demands, generate_story_drift_plots,
                                 check_beam_strength)

# ============================================================================
# CREATE MODEL
# ============================================================================
beam_ref_Ix_params = [1155.4185495416898, 0.020718421523148494]

bm_sections, col_sections = build_model(beam_ref_Ix_params, not_for_optimization=True)

# opsv.plot_model(node_labels=0, element_labels=0)

# ============================================================================
# GRAVITY & EIGEN ANALYSIS
# ============================================================================
angular_freq, periods, modal_prop = run_eigen_gravity_analysis(not_for_optimization=True)

# ==========================================================================
# ELF ANALYSIS
# ==========================================================================
story_weights, elf_push_pattern, elf_base_shear, elf_col_demands = run_elf_analysis(periods, 'modal', not_for_optimization=True)

# ============================================================================
# MRSA
# ============================================================================
mrsa_com_dispX, mrsa_com_dispY, elf_mrsaX_scale_factor, elf_mrsaY_scale_factor = run_mrsa(angular_freq, elf_base_shear, not_for_optimization=True)

# Clear model
ops.wipe()

# ============================================================================
# PDelta analysis & compute drifts
# ============================================================================
pdelta_method = 'B'
story_driftX, story_driftY, pdelta_factor = run_pdelta_analysis(beam_ref_Ix_params, pdelta_method, angular_freq, periods, story_weights,
                                                    mrsa_com_dispX, mrsa_com_dispY, elf_base_shear, elf_push_pattern,
                                                    elf_mrsaX_scale_factor, elf_mrsaY_scale_factor,
                                                    results_root_folder='./', not_for_optimization=True)

# Beam & Column demands
beam_demands_X, beam_demands_Y, col_demands_X, col_demands_Y = get_mrsa_and_torsional_demands(angular_freq,
                                                                                              pdelta_factor, elf_mrsaX_scale_factor,
                                                                                              elf_mrsaY_scale_factor, './mrsa_results',
                                                                                              './accidental_torsion_results')

# Check beam strength
check_beam_strength(bm_sections, beam_demands_X, beam_demands_Y)

# Check column strength


# Plot story drifts
generate_story_drift_plots(pdelta_method, story_driftX, story_driftY)

