# -*- coding: utf-8 -*-
"""
Simple simulations
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
import math
from scipy import stats
from bitarray import bitarray
import pickle
import community
from bitarray import bitarray
import time
import copy
import datetime

np.random.seed()

from communication_network import *

global b, N, payoff, K

def run_network_game(network):
    """update node payoffs"""
    for u in network.nodes() :
        network.node[u]['payoff'] = np.mean(network.node[u]['rewards'])



def accum_payoff(network, u):
    """return the acumulated payoff of node u"""
    global payoff
    p = 0
    s_u = network.node[u]['strategy']
    for v in network.node[u]['neighbors'] :
        s_v = network.node[v]['strategy']
        p += payoff[s_u][s_v]
    return p



def sim(network,field,n_mix=10000,n_data=1000):
    """run the simulation for n_mix+n_data times and return data for last n_data iterations,
    collect data on """
    global K
    data = [bitarray([False for x in network.nodes()]) for y in range(int(n_data))]
    for i in range(n_mix):
        information_sharing(network)
        arm_selection(network,field)
    for j in range(n_data):
        information_sharing(network)
        arm_selection(network,field)
        record_data_sharing(network,data[j])
    return data
    
    
def record_data(network, data_array) :
    """take single iteration snapshot of the rate of communication"""
    for n in network.nodes() :
        data_array[n] = np.len(network.node[n]['information shared with'])
        

        
def record_data_sharing(network, data_array) :
    """take single iteration snapshot of the mean rewards"""
    for n in network.nodes() :
        data_array[n] = np.sum(network.node[n]['rewards'])
            
    
def sim_group_size(network,arms, n_time=1000, alg=discounted_thompson):
    for i in range(1,n_time):
        information_sharing_thompson(network)
        arm_selection(network,arms, alg)
    data = np.zeros(len(network))
    record_data_sharing(network,data)
    return np.mean(data)/n_time


def full_sim_group_size(comm_prop, arms, n_time=1000, alg=discounted_thompson, gamma = 1, comm_init=False):
    """Full simulation - NOTE: ADD VARIOUS VALUES OF THE PROPENSITY TO SHARE INFORMATION"""
    global K
    G = initialise_network_arms(K, arms, alg, gamma,comm_init)
    data = []
    for i in range(2,K+1):
        H = copy.deepcopy(G)
        network = H.subgraph([j for j in range(i)])
        init(network)
        add_weights_complete_graph(network, comm_prop, const = True)
        #print (sim_group_size(network,n_time))
        data.append(sim_group_size(network,arms,n_time,alg))
    return data
    

# get data for for each propensity to share
def get_simulation_data(n_time=1000, n_steps = 10, alg=discounted_thompson, arm_type="bernoulli", comm_init=False) :
    global K
    data = dict()
    if arm_type == "bernoulli":
        arm_field = bernoulli_arms(K)
    elif arm_type == "gaussian":
        arm_field = truncated_gaussian_arms(K)
    for i in range(n_steps+1) :
        #print(str(i/float(n_steps)*100)+'%')
        data[i/float(n_steps)*100] = np.array(full_sim_group_size(i/float(n_steps),arm_field, n_time, alg, comm_init))
    return data

    
# take multiple simulations and take averages of the obtained data
def simulations(n_iter = 1000, n_time=1000, n_steps = 10, alg=discounted_thompson, arm_type="bernoulli", comm_init = False):
    global K
    data = dict()
    for k in range(n_iter):
        print(str(k/float(n_iter)*100)+'%')
        sim_data = get_simulation_data(n_time, n_steps, alg, arm_type, comm_init)
        for i in range(n_steps+1):
            data[i/float(n_steps)*100] = (k*data.get(i/float(n_steps)*100,np.zeros(K-1)) + sim_data[i/float(n_steps)*100])/(k+1)
    return data
    
    
# plots the degree distribution
def plot_results(n_iter = 1000, n_time=1000, n_steps = 20, alg=discounted_thompson, arm_type="bernoulli", plot_title='',comm_init=False ) :
    global K 
    sim_data = simulations(n_iter, n_time, n_steps, alg, arm_type, comm_init)
    grp_size = [i for i in range(2,K+1)]
    plt.figure()
    for key,value in sim_data.items():
        #print (key,value)
        plt.plot(grp_size, value, label=key)
    plt.xlabel('Group size')
    plt.ylabel('Average pay-off')
    plt.title(plot_title)
    plt.ylim(0.5,0.9)
    plt.xlim(1,K+20)
    plt.legend(loc="right", title="Propensity to share")
    plt.show()
    

#############################################################################
# progress messages
global pm
pm = True


def p_message(message) :
  global pm
  if pm == True :
    print(message)
    
    
def sec_to_string(seconds) :
  return str(datetime.timedelta(seconds=seconds))
    
    
# run a full simulation for with specified (networkx) graph constructor
# and specified parameters
def full_sim_run(n_iter = 1000, n_time=1000, n_steps = 20,algs = [discounted_thompson,discounted_thompson_general], arms=["bernoulli","gaussian"], sims_per_alg=1, params = None, comm_init=False) :
    simulation = dict()
    assert len(algs)==len(arms), "Error, the number of algorithms is different from the number of arms."
    simulation['algs'] = algs
    simulation['arms'] = arms
    simulation['params'] = params
    simulation['sims_per_alg'] = sims_per_alg
    simulation['n_iter'] = n_iter
    simulation['n_time'] = n_time
    simulation['n_steps'] = n_steps
    #simulation['data'] = []          TO-DO: CHANGE IN CASE WE WANT TO RETREIVE THE SIMULATION DATA
    start = int(round(time.time()))
    for i in range(len(algs)) :
        p_message('algorithm '+str(i+1)+' of '+str(len(algs))+'algorithms: begin')
        for j in range(sims_per_alg) :
            p = (i*sims_per_alg + j)/float(len(algs)*sims_per_alg)
            elapsed = int(round(time.time())) - start
            if (p != 0) :
                remaining = int(elapsed*((1-p)/float(p)))
            else :
                remaining = 0
            p_message('elapsed time: ' + sec_to_string(elapsed) + ' |   remaining: ' + sec_to_string(remaining))
            p_message('algorithm '+str(i+1)+' of '+str(len(algs))+': sim '+str(j+1)+' of '+str(sims_per_alg)+': running')
            plot_results(n_iter, n_time, n_steps, algs[i], arms[i], algs[i].__name__+'/'+arms[i], comm_init)
    elapsed = int(round(time.time())) - start
    p_message('total time taken: ' + sec_to_string(elapsed))
    #return simulation
    
    
##############################################################################
##############################################################################
K = 50
#print (get_simulation_data(100,3))
#print (simulations(10,100,3))

full_sim_run(100,50,5)

#algs = [discounted_thompson,discounted_thompson_general]
#print (algs[0].__name__)