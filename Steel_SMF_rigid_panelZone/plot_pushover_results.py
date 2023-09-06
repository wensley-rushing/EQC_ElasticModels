# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:24:48 2023

@author: udu
"""

# import os
# import glob
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter

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

# Import pushover results
base_shearX = np.loadtxt('./pushover/baseShearX.txt')
base_shearY = np.loadtxt('./pushover/baseShearY.txt')

total_base_shearX = -base_shearX.sum(axis=1)
total_base_shearY = -base_shearY.sum(axis=1)

COM_dispX = np.loadtxt('./pushover/COM_dispX.txt')
COM_dispY = np.loadtxt('./pushover/COM_dispY.txt')

story_driftX = np.zeros_like(COM_dispX)
story_driftY = np.zeros_like(COM_dispY)

for ii in range(COM_dispX.shape[1]):
    if ii == 0:
        story_driftX[:, ii] = COM_dispX[:, ii] / flr1 * 100
        story_driftY[:, ii] = COM_dispY[:, ii] / flr1 * 100
    else:
        story_driftX[:, ii] = (COM_dispX[:, ii] - COM_dispX[:, ii-1]) / typ_flr_height * 100
        story_driftY[:, ii] = (COM_dispY[:, ii] - COM_dispY[:, ii-1]) / typ_flr_height * 100


design_shear_force = 8443 * kN
# ============================================================================
# Generate story drift profile at the timestep where the base shear = design shear force
# ============================================================================
design_shear_force = 8443 * kN

# Get the index where the total base shear is approximately equal to the design base shear
ind_design_shear = np.where(total_base_shearX >= 8443)[0][0]

# Extract the story drift profile at the timestep (index) obtained above
story_driftX_at_design_shear = story_driftX[ind_design_shear, :]
story_driftY_at_design_shear = story_driftY[ind_design_shear, :]

# Generate plot of story drift profile at design shear
fig, ax = plt.subplots(figsize=(4.5, 7.5), constrained_layout=True)
fig.suptitle('Story drift profile at design shear force, V = {:,.0f} kN'.format(design_shear_force), fontdict=title_font)

ax.plot(story_driftX_at_design_shear, elev, color='#f4a582')
ax.plot(story_driftX_at_design_shear, elev, color='#f4a582', marker='o')

ax.set_xlabel('Story drift [%]', fontdict=axes_font)
ax.set_ylabel('Elevation [m]', fontdict=axes_font)

plt.savefig('SSMF_storyDriftProfile2.pdf', dpi=1200)

# ax.set_yticks(['Lvl 1', 'Lvl 2', 'Lvl 3', 'Lvl 4', 'Lvl 5', 'Lvl 6',
#                'Lvl 7', 'Lvl 8', 'Lvl 9', 'Lvl 10', 'Roof'])

ax.set_yticks(elev)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

ax.set_xlim(0.0)
ax.set_ylim(0.0)
ax.tick_params(axis='both', direction='in', colors='grey', labelcolor='grey', zorder=3.0, labelsize=8.0)

'''
roof_height = 35.5
roof_drift = roof_COM_dispX/roof_height * 100


fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
fig.suptitle('Pushover plot', fontdict=title_font)
ax.plot(roof_drift, total_base_shearX)

ax.set_xlim(0, roof_drift.max())
ax.set_ylim(0, total_base_shearX.max())

ax.set_xlabel('Roof drift [%]', fontdict=axes_font)
ax.set_ylabel('Base shear [kN]', fontdict=axes_font)

ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.tick_params(axis='both', direction='in', colors='grey', labelcolor='grey', zorder=3.0, labelsize=8.0)

plt.savefig('RCSW_pushover.pdf', dpi=1200)
'''
