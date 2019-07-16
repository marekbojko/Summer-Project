# Summer Project

Run Thompson/simple_sims to run a simulation of agents solving Bayesian multi-armed bandits and sharing information 

Note: I still haven't managed to install Jupyter notebook. The kernel kept dying and so I updates anaconda. Now I'm unable to download Jupyter again. I'll try to resolve the issue asap.

To see an animation with a low number of iterations, run:

from evolving_comm_network import *

PD = Simulation_PD(25, 80, bernoulli_arms, discounted_thompson, 10)
PD.mult_generations()
