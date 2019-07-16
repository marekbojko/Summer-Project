# -*- coding: utf-8 -*-
"""
Evolutionary algorithms
"""

import networkx as nx
import numpy as np
import random

np.random.seed(None)

class roulette_wheel(object):
    
    def __init__(self,network):
        self.pop = network
        self.pop_fitness = self.pop_fitness(network)
        self.probabilities = self.get_probability_list()
    
    def pop_fitness(self,network):
        pop_fitness = dict()
        for n in network.nodes():
            pop_fitness[n] = network.node[n]['accumulated payoff']
        return pop_fitness
        
    def get_probability_list(self):
        fitness = self.pop_fitness.values()
        total_fit = sum(fitness)
        relative_fitness = [f/total_fit for f in fitness]
        probabilities = [sum(relative_fitness[:i+1]) 
                         for i in range(len(relative_fitness))]
        return probabilities


    def roulette_wheel_pop(self):
        chosen = []
        for n in range(len(self.pop)):
            r = random.random()
            for i, individual in enumerate(self.pop.nodes()):
                if r <= self.probabilities[i]:
                    chosen.append(individual)
                    break
        return chosen