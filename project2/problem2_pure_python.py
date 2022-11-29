import numpy as np
import matplotlib.pyplot as plt
import import_ipynb
from worm_simulation import Worm_algorithm
import pandas as pd

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

### Get the square of the winding number varying the beta ###
for beta in beta_array:
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
        
        #print('-----------------' + str(counter) + ' New grid created----------------------')
        #if (((counter + 1) % 500) == 0): print('-----------------' + str(counter + 1) + ' New grid created----------------------')
    
    susceptibility_array.append(np.average(num_winding_square_Monte_Carlo) / (N * beta))
    data = np.array([num_particles_Monte_Carlo, energy_Monte_Carlo, num_winding_square_Monte_Carlo]).transpose()
    pd.DataFrame(data).to_csv("beta_"+str(beta)+"_test.csv", index=False, header=["n", "e_tilda", "w^2"])
    print('-----------------------beta = %.3f done-----------------------------' % beta)

