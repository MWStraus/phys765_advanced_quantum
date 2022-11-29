import numpy as np
import matplotlib.pyplot as plt
from worm_simulation_pure_python import Worm_algorithm
import pandas as pd
from pathos.multiprocessing import Pool
import time

################################################### Block function #####################################################

################################################### Simulation #####################################################

### Parameter Setting ###
[N, mu, epsilon, dimension] = [20, 1.0, 0.01, 3]

N_beta = 15
beta_min = 0.75
beta_max = 1.1
beta_array = np.linspace(beta_min, beta_max, N_beta)

susceptibility_array = []
susceptibility_error_array = []

num_trials = 50# Monte-Carlo try number
block_size = 50

def simulation(beta):
    parameters = [N, mu, epsilon, beta, dimension]
    worm_algorithm = Worm_algorithm(parameters) # Initialization
    num_winding_square_Monte_Carlo = [] # Saves the winding number result of each Monte-Carlo try
    num_particles_Monte_Carlo = []
    energy_Monte_Carlo = []
    # Monte-Carlo simulation part
    for counter in range(num_trials):
        num_particles, e_tilda, weight, num_winding = worm_algorithm.measure_observables()
        num_particles_Monte_Carlo.append(num_particles)
        energy_Monte_Carlo.append(e_tilda)
        num_winding_square_Monte_Carlo.append(num_winding/3)

        old_grid = worm_algorithm.get_pathed_grid()
        new_grid = worm_algorithm.update_grid(old_grid)
        worm_algorithm.set_pathed_grid(new_grid)

    susceptibility_array.append(np.average(num_winding_square_Monte_Carlo) / (N * beta))
    data = np.array([num_particles_Monte_Carlo, energy_Monte_Carlo, num_winding_square_Monte_Carlo]).transpose()
    pd.DataFrame(data).to_csv("beta_"+str(beta)+"_test_parallel.csv", index=False, header=["n", "e_tilda", "w^2"])
    print('-----------------------beta = %.3f done-----------------------------' % beta)
    return None

### Get the square of the winding number varying the beta ###
if __name__ == "__main__":
    start_time = time.perf_counter()
    with Pool() as pool:
        blank = pool.map(simulation, beta_array)
    finish_time = time.perf_counter()
    print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
    print("---")

    start_time = time.perf_counter()
    for beta in beta_array:
        blank = simulation(beta)
    finish_time = time.perf_counter()
    print("Program finished in {} seconds".format(finish_time-start_time))

