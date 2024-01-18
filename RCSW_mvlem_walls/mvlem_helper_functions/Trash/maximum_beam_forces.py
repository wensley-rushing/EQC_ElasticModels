# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:16:25 2023

@author: Uzo Uwaoma - udu@uw.edu
"""


def get_max_shear_and_moment(peak_total_resp):
    """
    This functions gives the maximum shear and bending moment for all beams ona floor.

    It takes in the peak total response from a MRSA. Then indexes into the shear force
    and moment values, taking cognizance of how OpenSeesPy recorders for beam-column
    elements force response are stored.

    In a 3D model each node of a beam-column element has 6DOFs, so OpenSeesPy
    records the underlisted force response for each element:
    Fx-i, Fy-i, Fz-i, Mx-i, My-i, Mz-i, Fx-j, Fy-j, Fz-j, Mx-j, My-j, Mz-j

    Parameters
    ----------
    peak_total_resp : Numpy array
        Peak total response for all beam elements in a floor.

    Returns
    -------
    max_shear_force : float
        Maximum beam shear force on a floor.
    max_mom : float
        Maximum beam bending moment on a floor.

    """

    # Extract shear forces and moments
    shear_force = peak_total_resp[2::6]

    mom_x = peak_total_resp[3::6]
    mom_y = peak_total_resp[4::6]
    mom_z = peak_total_resp[5::6]

    max_shear_force = shear_force.max()
    max_mom = max(mom_x.max(), mom_y.max(), mom_z.max())

    return max_shear_force, max_mom
