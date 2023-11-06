# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:18:37 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import numpy as np
from scipy.optimize import fsolve

# Define Units
sec = 1

# Metric Units
m = 1
kN  = 1

mm = m / 1000
N = kN/1000
kPa = 0.001 * N/mm**2   # Kilopascal
MPa = 1 * N/mm**2       # Megapascal
GPa = 1000 * N/mm**2    # Gigapascal
grav_metric = 9.81 * m/sec**2

reinf_spac = 200 * mm  # Center-to-center spacing of wall longitudinal reinforcement


def get_wall_reinf_ratio(lw, tw, wall_P, wall_M, fpc, fy, steelE):


    def wall_steel_spacing_ratio(wall_reinf, wall_length=lw, wall_thick=tw, wall_load=wall_P,
                                 wall_moment=wall_M, conc_fpc=fpc, steel_fy=fy, steel_mod=steelE):

        # Material properties

        # print(wall_load, wall_moment)
        beta = 0.85

        alpha = steel_fy /(0.003 * steel_mod)

        depth_neutral_axis = ((wall_reinf * steel_fy * wall_length +  wall_load) /
                              (2 * wall_reinf * steel_fy + 0.85 * conc_fpc * beta * wall_thick))

        param1 = wall_load * 0.5 * wall_length
        param2 = 0.425 * conc_fpc * beta**2 * depth_neutral_axis**2 * wall_thick
        param3 = wall_reinf * 0.5 * steel_fy * (2*(1 + (alpha**2)/3)*depth_neutral_axis**2 - wall_length**2)

        sys_of_eqn = param1 - param2 - param3 - wall_moment

        return sys_of_eqn

    # Solve for value of As/s (coded as `wall_reinf` in above equations) which satisfies equilibrium
    steel_area_to_spacing_ratio = fsolve(wall_steel_spacing_ratio, [1], args=(lw, tw, wall_P, wall_M, fpc, fy, steelE))[0]
    # print(steel_area_to_spacing_ratio)

    # Calculate reinforcement ratio corresponding to obtained As/s value
    wall_reinf_ratio = steel_area_to_spacing_ratio / tw

    # Compute minimum reinforcement ratio for longitudinal reinforcement
    # Values of fpc & fy must be in MPa below. Hence, we need to conver back.
    # min_reinf_ratio = np.sqrt(fpc/1000) / (4 * fy / 1000) #  NZS 3101.1.2006: Sect 11.4.4.2

    return wall_reinf_ratio

    # if wall_reinf_ratio < 0:
    #     return min_reinf_ratio
    # else:
    #     return wall_reinf_ratio


if __name__ == "__main__":

    lw = 7.550 * m
    tw = 300 * mm
    wall_P = 572.993 * kN
    wall_M = 13446 * kN * m
    fpc = 40 * MPa
    fy = 500 * MPa
    steelE = 200 * GPa

    root = get_wall_reinf_ratio(lw, tw, wall_P, wall_M, fpc, fy, steelE)
