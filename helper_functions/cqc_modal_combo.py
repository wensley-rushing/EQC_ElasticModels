# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:07:11 2023

@author: Uzo Uwaoma - udu@uw.edu
"""


import numpy as np


def modal_combo(peak_modal_resp, ang_freq, damp, num_modes):
    """
    Obtains peak modal response using the CQC rule.

    Note: This function assumes that uniform damping is applied to all modes.

    Parameters
    ----------
    peak_modal_resp : Numpy Array
        m x n vector of peak modal response for a given response of interest.
        where,
            m: Number of modes consisdered
            n: Number of DOFs of response per element
    ang_freq : Numpy array
        m x 1 vector of natural angular frequencies. i.e, omega values
    damp : float
        Amount of damping applied to structural system
    num_modes : int
        Number of modes to use in computing peak modal response.

    Returns
    -------
    Numpy array
        n x 1 vector of peak modal response.

    """
    peak_total_resp = 0.

    for ii in range(num_modes):
        for jj in range(num_modes):
            omega_i = ang_freq[ii]
            omega_j = ang_freq [jj]
            beta_in = omega_i / omega_j

            correl_coeff = ((8 * damp**2 * (1 + beta_in) * beta_in**(3/2)) /
                            ((1-beta_in**2)**2 + 4*damp**2*beta_in*(1+beta_in)**2))

            peak_total_resp += (correl_coeff * peak_modal_resp[ii] * peak_modal_resp[jj])

    return np.sqrt(peak_total_resp)
