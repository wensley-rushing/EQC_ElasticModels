# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:07:11 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import sys
import numpy as np

# Append directory of helper functions to Pyhton Path
sys.path.append('./')

from helper_functions.cqc_modal_combo import modal_combo


def compute_story_drifts(com_disp, story_heights, ang_freq, damp, num_modes):

    # Obtain peak total response
    peak_tot_disp = modal_combo(com_disp, ang_freq, damp, num_modes)

    # Compute inter story displcaments
    peak_inter_story_disp = np.insert(np.diff(peak_tot_disp), 0, peak_tot_disp[0])

    # Compute inter story drift ratio
    story_drift_ratio = peak_inter_story_disp / story_heights * 100

    return story_drift_ratio



# Compute inter story drift ratio
# story_drift_ratio = peak_tot_disp / story_heights * 100
