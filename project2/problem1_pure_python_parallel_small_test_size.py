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
        #if counter%int(num_trials/1000)==0:
        #    print("mu:", mu, "--", round(counter/num_trials*100, 3),"%")
    susceptibility_array.append(np.average(num_winding_square_Monte_Carlo) / (N * beta))
    print('-----------------------mu = %.3f done-----------------------------' % mu)
    data = np.array([np.ones(len(num_particles_Monte_Carlo))*mu, num_particles_Monte_Carlo, energy_Monte_Carlo, num_winding_square_Monte_Carlo]).transpose()
    return data

### Get the square of the winding number varying the beta ###
if __name__ == "__main__":
    print("Starting")
    ### Parameter Setting ###
    [N, mu, epsilon, dimension] = [4, 1.0, 0.01, 3]
    N_mu = 3
    mu_min = 0
    mu_max = 0.5
    beta = 100
    mu_array = np.linspace(mu_min, mu_max, N_mu)

    num_trials = 100# Monte-Carlo try number
    print("got parameters")
    all_parameters = []
    for mu in mu_array:
        parameters = [ N, mu, epsilon, beta, dimension, num_trials]
        all_parameters.append(parameters)
    print("Starting Simulation")
    all_data = []
    for jj in range(3):
        with Pool() as pool:
            data = pool.map(simulation, all_parameters[jj:jj+1])
        all_data.append(data)
    data2 = np.reshape(all_data, (N_mu*num_trials, 4))
    print("----------data------------------")
    print(data2)
    print("writing data")
    pd.DataFrame(data2).to_csv("problem1data_small_test.csv", index=False,
                              header=["mu", "n", "e_tilda", "w^2"])


