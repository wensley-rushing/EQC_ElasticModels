# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:07:11 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import sys
# import pandas as pd

# Append directory of helper functions to Pyhton Path
sys.path.append('./')

from helper_functions.cqc_modal_combo import modal_combo


def compute_story_drifts(com_disp, story_heights, ang_freq, damp, num_modes):

    # Obtain peak total response
    peak_tot_disp = modal_combo(com_disp, ang_freq, damp, num_modes)

    story_drift = peak_tot_disp / story_heights * 100

    return story_drift


"""
def compute_story_drifts(com_dispX, com_dispY, story_heights, ang_freq, damp, num_modes):

    # Obtain peak total response
    peak_tot_dispX = modal_combo(com_dispX, ang_freq, damp, num_modes)
    peak_tot_dispY = modal_combo(com_dispY, ang_freq, damp, num_modes)

    story_driftX = peak_tot_dispX / story_heights * 100
    story_driftY = peak_tot_dispY / story_heights * 100

    story_drift = pd.DataFrame()
    story_drift['Floor'] = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']

    story_drift['DriftX (%)'] = story_driftX
    story_drift['DriftY (%)'] = story_driftY

    return story_drift
"""
