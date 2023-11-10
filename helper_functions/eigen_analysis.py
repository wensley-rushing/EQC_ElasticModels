# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:54:46 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import numpy as np


def run_eigen_analysis(ops, num_modes, damping_ratio, report_folder, lfre_system):

    ops.wipeAnalysis()

    eigen_values = ops.eigen(num_modes)
    angular_freq = [np.sqrt(lam) for lam in eigen_values]
    nat_freq = [omega/(2*np.pi) for omega in angular_freq]
    periods = [1/freq for freq in nat_freq]

    print('')
    for ii in range(1, num_modes+1):
        print('Mode {} Tn: {:.3f} sec'.format(ii, periods[ii-1]))

    modal_prop = ops.modalProperties('-file', report_folder + 'ModalReport_' + lfre_system + '.txt', '-unorm', '-return')

    # Apply Damping
    # Mass and stiffness proportional damping will be applied

    mass_prop_switch = 1.0
    stiff_curr_switch = 1.0
    stiff_comm_switch = 0.0  # Last committed stiffness switch
    stiff_init_switch = 0.0  # Initial stiffness switch

    # Damping coeffieicent will be compusted using the 1st & 5th modes
    omega_i = angular_freq[0]  # Angular frequency of 1st Mode
    omega_j = angular_freq[4]  # Angular frequency of 5th Mode

    alpha_m = mass_prop_switch * damping_ratio * ((2*omega_i*omega_j) / (omega_i + omega_j))
    beta_k = stiff_curr_switch * damping_ratio * (2 / (omega_i + omega_j))
    beta_k_init = stiff_init_switch * damping_ratio * (2 / (omega_i + omega_j))
    beta_k_comm = stiff_comm_switch * damping_ratio * (2 / (omega_i + omega_j))

    ops.rayleigh(alpha_m, beta_k, beta_k_init, beta_k_comm)

    return angular_freq, modal_prop
