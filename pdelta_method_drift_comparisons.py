# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:22:42 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter

# Define Units
sec = 1

# US units
inch = 1
kips = 1

ft = 12 * inch
lb = kips/1000
ksi = kips/inch**2
psi = ksi/1000
grav_US = 386.4 * inch/sec**2

# Metric Units
m = 1
kN  = 1

mm = m / 1000
N = kN/1000
kPa = 0.001 * N/mm**2   # Kilopascal
MPa = 1 * N/mm**2       # Megapascal
GPa = 1000 * N/mm**2    # Gigapascal
grav_metric = 9.81 * m/sec**2

# Set plotting parameters
mpl.rcParams['axes.edgecolor'] = 'grey'
mpl.rcParams['lines.markeredgewidth'] = 0.4
mpl.rcParams['lines.markeredgecolor'] = 'k'
plt.rcParams.update({'font.family': 'Times New Roman'})

axes_font = {'family': "sans-serif",
              'color': 'black',
              'size': 8
              }

title_font = {'family': 'sans-serif',
              'color': 'black',
              'weight': 'bold',
              'size': 8}

legend_font = {'family': 'Times New Roman',
              'size': 8}

# Floor elevations
typ_flr_height = 3.1 * m
ground_flr = 0.0 * m

flr1 = 4.5 * m
flr2 = flr1 + typ_flr_height
flr3 = flr2 + typ_flr_height
flr4 = flr3 + typ_flr_height
flr5 = flr4 + typ_flr_height
flr6 = flr5 + typ_flr_height
flr7 = flr6 + typ_flr_height
flr8 = flr7 + typ_flr_height
flr9 = flr8 + typ_flr_height
flr10 = flr9 + typ_flr_height
roof_flr = flr10 + typ_flr_height

elev = [flr1, flr2, flr3, flr4, flr5, flr6, flr7, flr8, flr9, flr10, roof_flr]

def generate_drift_plots(drift_results_direc):

    # Load in Story drifts
    story_driftX_pDel_A = np.loadtxt(drift_results_direc + 'driftX-PDeltaMethodA.txt')
    story_driftY_pDel_A = np.loadtxt(drift_results_direc + 'driftY-PDeltaMethodA.txt')

    story_driftX_pDel_B = np.loadtxt(drift_results_direc + 'driftX-PDeltaMethodB.txt')
    story_driftY_pDel_B = np.loadtxt(drift_results_direc + 'driftY-PDeltaMethodB.txt')

    fig, ax = plt.subplots(1, 2, figsize=(6.0, 7.5), sharey=True, constrained_layout=True)
    fig.suptitle('Story drift ratios using PDelta methods A & B', fontdict=title_font)

    ax[0].vlines(story_driftX_pDel_A[0], 0.0, elev[0], color='#f1a340', label='Method A')
    ax[1].vlines(story_driftY_pDel_A[0], 0.0, elev[0], color='#f1a340')

    ax[0].vlines(story_driftX_pDel_B[0], 0.0, elev[0], color='#998ec3', label='Method B')
    ax[1].vlines(story_driftY_pDel_B[0], 0.0, elev[0], color='#998ec3')


    for ii in range(1, len(elev)):
        # P-Delta method A
        ax[0].hlines(elev[ii-1], story_driftX_pDel_A[ii-1], story_driftX_pDel_A[ii], color='#f1a340')
        ax[0].vlines(story_driftX_pDel_A[ii],  elev[ii-1], elev[ii], color='#f1a340')

        ax[1].hlines(elev[ii-1], story_driftY_pDel_A[ii-1], story_driftY_pDel_A[ii], color='#f1a340')
        ax[1].vlines(story_driftY_pDel_A[ii],  elev[ii-1], elev[ii], color='#f1a340')

        # P-Delta method B
        ax[0].hlines(elev[ii-1], story_driftX_pDel_B[ii-1], story_driftX_pDel_B[ii], color='#998ec3')
        ax[0].vlines(story_driftX_pDel_B[ii],  elev[ii-1], elev[ii], color='#998ec3')

        ax[1].hlines(elev[ii-1], story_driftY_pDel_B[ii-1], story_driftY_pDel_B[ii], color='#998ec3')
        ax[1].vlines(story_driftY_pDel_B[ii],  elev[ii-1], elev[ii], color='#998ec3')


    ax[0].set_title('X - Direction', fontsize=12, family='Times New Roman')
    ax[1].set_title('Y- Direction', fontsize=12, family='Times New Roman')

    ax[0].set_ylabel('Story elevation (m)', fontdict=axes_font)
    ax[0].legend(loc='upper left', fontsize=12, prop=legend_font, frameon=True, handlelength=2.0)


    for axx in ax.flat:
        axx.set_xlim(0.0)
        axx.set_ylim(0.0, elev[-1])

        axx.grid(True, which='major', axis='both', ls='-.', linewidth=0.6)

        axx.set_yticks(elev)

        axx.set_xlabel('Story drift ratio (%)', fontdict=axes_font)

        axx.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
        axx.tick_params(axis='both', direction='in', colors='grey', labelcolor='grey', zorder=3.0, labelsize=8.0)

    plt.savefig(drift_results_direc + 'DriftPlots-PDeltaMethodComparisons.png', dpi=1200)


generate_drift_plots('./Steel_SMF_rigid_panelZone/')
generate_drift_plots('./RCSW_rigid_wall_links/')
