import numpy as np
import matplotlib.pyplot as plt
from worm_simulation_pure_python import Worm_algorithm
import pandas as pd
from pathos.multiprocessing import Pool
import time

################################################### Block function #####################################################

################################################### Simulation #####################################################


susceptibility_array = []
susceptibility_error_array = []

def simulation(parameters):
    N, mu, epsilon, beta, dimension, num_trials = parameters
    worm_parameters = [N, mu, epsilon, beta, dimension]
    worm_algorithm = Worm_algorithm(worm_parameters) # Initialization
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
        if counter%int(num_trials/1000)==0:
            print("beta:", beta, "--", round(counter/num_trials*100, 3),"%")

    susceptibility_array.append(np.average(num_winding_square_Monte_Carlo) / (N * beta))
    data = np.array([num_particles_Monte_Carlo, energy_Monte_Carlo, num_winding_square_Monte_Carlo]).transpose()
    pd.DataFrame(data).to_csv("problem3data\\N_"+str(N)+"_beta_"+str(beta)+"_parallel.csv", index=False, header=["n", "e_tilda", "w^2"])
    print('-----------------------beta = %.3f done-----------------------------' % beta)
    return None

### Get the square of the winding number varying the beta ###
if __name__ == "__main__":
    
### Parameter Setting ###
    [N, mu, epsilon, dimension] = [20, 1.0, 0.01, 3]
    N_beta = 8
    beta_min = 0.88
    beta_max = 0.95
    beta_array = np.linspace(beta_min, beta_max, N_beta)

    num_trials = 50000# Monte-Carlo try number
    all_parameters = []
    for beta in beta_array:
        parameters = [ N, mu, epsilon, beta, dimension, num_trials]
        all_parameters.append(parameters)
    print("Starting Simulation")
    with Pool() as pool:
        blank = pool.map(simulation, all_parameters)


