# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 00:47:51 2024

@author: Uzo Uwaoma - udu@uw.edu
"""

import sys
import time
import random
import numpy as np
import pandas as pd
import opensees.openseespy as ops


# Append directory of helper functions to Python Path
sys.path.append('./')

from generate_ssmf_model import (build_model, run_eigen_gravity_analysis,
                                 run_elf_analysis, run_mrsa, run_pdelta_analysis,
                                 get_mrsa_and_torsional_demands)

# Initialize array of possible values for beam Ix
nzs_beams = pd.read_excel('../../../nzs_steel_database.xlsx', sheet_name='Beams',
                               index_col='Designation')

bm_Ix_vals = np.array(list(nzs_beams['Ix']))
bm_Zx_vals = np.array(list(nzs_beams['Zx']))

# Maximum beam section will be a function of Zx to enable capacity design
max_bm_Zx = bm_Zx_vals.max() / (1.25 * 2) # (Divided by (1.25*2) so there is an available column in the event that 2 beams framing into a column reach Mp.)

# Define bounds on possible values for first floor beam Ix.
bm_Ix_min = bm_Ix_vals.min()
bm_Ix_max = bm_Ix_vals[bm_Zx_vals <= max_bm_Zx][0]
bm_Ix_bounds = (bm_Ix_min, bm_Ix_max)

# Define bounds on slope of Ix distribution
bm_Ix_slope_min = 1 / 310
bm_Ix_slope_max = 1 / 31
bm_Ix_slope_bounds = (bm_Ix_slope_min, bm_Ix_slope_max)


def objective_func(optim_params):

    # Build model
    build_model(optim_params)

    # Run eigen analysis
    angular_freq, periods = run_eigen_gravity_analysis()

    story_weights, elf_push_pattern, elf_base_shear = run_elf_analysis(periods, 'modal')

    # Perform MRSA
    mrsa_com_dispX, mrsa_com_dispY, elf_mrsaX_scale_factor, elf_mrsaY_scale_factor = run_mrsa(angular_freq, elf_base_shear)
    ops.wipe()

    # PDelta Analysis
    pdelta_method = 'B'
    _, max_theta = run_pdelta_analysis(optim_params, pdelta_method, angular_freq, periods, story_weights,
                                                    mrsa_com_dispX, mrsa_com_dispY, elf_base_shear, elf_push_pattern,
                                                    elf_mrsaX_scale_factor, elf_mrsaY_scale_factor,
                                                    results_root_folder='./optimization_results/')

    # # Beam & Column demands
    # beam_demands_X, beam_demands_Y, col_demands_X, col_demands_Y = get_mrsa_and_torsional_demands(angular_freq,
    #                                                                                               pdelta_factor, elf_mrsaX_scale_factor,
    #                                                                                               elf_mrsaY_scale_factor, './mrsa_results',
    #                                                                                               './accidental_torsion_results')

    return max_theta

'*********************************************************************************************'
'*********************************************************************************************'
'*********************************************************************************************'

init_time = time.time()

nv = 2  # number of variables
optim_prob = -1  # if minimization problem, optim_prob = -1; if maximization problem, optim_prob = 1

bounds = [bm_Ix_bounds, bm_Ix_slope_bounds]

particle_size = 50 * nv  # number of particles
# iterations = 1  # max number of iterations

w = 0.9   # inertia constant  0.75
c1 = 0.5  # cognitive constant
c2 = 2    # social constant

# END OF THE CUSTOMIZATION SECTION
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
if optim_prob == -1:
    initial_fitness = float("inf")  # for minimization problem
if optim_prob == 1:
    initial_fitness = -float("inf")  # for maximization problem
# -----------------------------------------------------------------------------


class Particle:
    def __init__(self, bounds):
        self.particle_position = []  # particle position
        self.particle_velocity = []  # particle velocity
        self.local_best_particle_position = []  # best position of the particle
        self.fitness_local_best_particle_position = initial_fitness  # initial objective function value of the best particle position
        self.fitness_particle_position = initial_fitness  # objective function value of the particle position

        for i in range(nv):
            self.particle_position.append(
                random.uniform(bounds[i][0], bounds[i][1]))  # generate random initial position
            self.particle_velocity.append(random.uniform(-1, 1))  # generate random initial velocity

    def evaluate(self, objective_func):
        self.fitness_particle_position = objective_func(self.particle_position)
        # print(self.fitness_particle_position)

        if optim_prob == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # update the local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best
        if optim_prob == 1:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # update the local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best

    def update_velocity(self, global_best_particle_position):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()

            cognitive_velocity = c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w * self.particle_velocity[i] + cognitive_velocity + social_velocity

    def update_position(self, bounds):  # Using a periodic boundary handler https://pyswarms.readthedocs.io/en/latest/api/pyswarms.backend.html
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]

            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][0] + ((self.particle_position[i] - bounds[i][1]) % np.abs(bounds[i][0] - bounds[i][1]))

            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][1] - ((bounds[i][0] - self.particle_position[i]) % np.abs(bounds[i][0] - bounds[i][1]))

global_best_theta_difference = initial_fitness
global_best_theta = initial_fitness
global_best_particle_position = []

# Build swarm
swarm = []
for i in range(particle_size):
    swarm.append(Particle(bounds))

fitness_history = []
param_history = []

tol = 1E-6
converg_count = 0

iter_count = 0
theta_limit = 0.3  # Limit on stability per NZS 1170.5:2004 - Sect 6.5.1

# for i in range(iterations):
while global_best_theta > theta_limit and global_best_theta_difference > 0.1:
    print("Iteration {}".format(iter_count+1))

    # cycle through particles in swarm and evaluate fitness
    for j in range(particle_size):
        print("Particle " + str(j+1))
        swarm[j].evaluate(objective_func)
        print("Theta: {:.3f}".format(swarm[j].fitness_particle_position))

        # determine if current particle is the best (globally)
        if optim_prob == -1:
            print("Best theta difference as at last iteration: {:.3f}".format(global_best_theta_difference))

            theta_diff = abs(swarm[j].fitness_particle_position - theta_limit)
            print("Theta difference for current iteration : {:.3f}".format(theta_diff))

            if theta_diff < global_best_theta_difference and swarm[j].fitness_particle_position < theta_limit:
                global_best_particle_position = list(swarm[j].particle_position)
                global_best_theta = float(swarm[j].fitness_particle_position)
                global_best_theta_difference = float(theta_diff)

        if optim_prob == 1:
            if swarm[j].fitness_particle_position > global_best_theta_difference:
                global_best_particle_position = list(swarm[j].particle_position)
                global_best_theta_difference = float(swarm[j].fitness_particle_position)

        print("New best theta diff: {:.3f}".format(global_best_theta_difference))
        print("")

   # cycle through swarm and update velocities and position
    for j in range(particle_size):
        swarm[j].update_velocity(global_best_particle_position)
        swarm[j].update_position(bounds)

    fitness_history.append(global_best_theta_difference)  # record the best fitness
    param_history.append(global_best_particle_position)  # record associated fitness parameters for each iteration

    print('iteration: {}, best_solution: {}, best_fitness: {}'.format(iter_count+1, global_best_particle_position,
                                                                      global_best_theta_difference))

    if iter_count > 0:
        if abs(fitness_history[-1] - fitness_history[-2]) < tol:
            converg_count += 1
        else:
            converg_count = 0

        if converg_count == 20:
            break

    print("After iter {}: ".format(iter_count+1), global_best_theta, global_best_theta_difference)
    print("")
    iter_count += 1

print('Optimal solution:', global_best_particle_position)
print('Objective function value:', global_best_theta_difference)

run_time = time.time() - init_time
print("\nRun time:  {} secs".format(run_time))

convergence_history = open("./optimization_results/SMF_nzs_theta.txt", 'w+')
convergence_history.write("Best Solution History: " + str(param_history) + "\n \n")
convergence_history.write("Best Fitness History: " + str(fitness_history) + "\n")
convergence_history.write("Run time: " + str(run_time) + " secs\n")
convergence_history.close()

