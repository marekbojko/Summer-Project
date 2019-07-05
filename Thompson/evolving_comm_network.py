# -*- coding: utf-8 -*-
"""
Evolving communication networks
"""

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
from mab_algorithms import *
from arms import *
from graph_measures import *

global K
K = 50
    
            
    
def sim_group_size(network,arms, n_time=1000, alg=discounted_thompson, gamma=1):
    oracle = np.argmax(np.array(arms.field_exp))
    oracle_performance = np.zeros(0)
    for i in range(1,n_time):
        network.run_round_sim(arms, alg, gamma)
        oracle_performance = np.append(oracle_performance,arms.field[oracle].draw_sample())
    data = np.zeros(len(network))
    record_data_sharing(network,data)
    return np.mean(data)/n_time/np.mean(oracle_performance),network


def full_sim_group_size(comm_prop, arms, n_time=1000, alg=discounted_thompson, gamma = 1, comm_init=False, const = True):
    """Full simulation - NOTE: ADD VARIOUS VALUES OF THE PROPENSITY TO SHARE INFORMATION"""
    G = nx.MultiDiGraph()
    A = comm_graph(G)
    A.initialise(arms.n, arms.n, alg, gamma, True, const, comm_prop, comm_init)
    A.select_arm(arms, alg, gamma)
    data = np.empty(0)
    reciprocity = np.empty(0)
    clique_numbers = np.empty(0)
    avg_clustering = np.empty(0)
    for i in range(2,K+1,5):
        H = copy.deepcopy(A)
        network = H.subgraph([j for j in range(i)])
        network.init_info_further(arms.n)
        network.init_sharing_vector(const,comm_prop)
        #print (sim_group_size(network,n_time))
        data = np.append(data,sim_group_size(network,arms,n_time,alg, gamma)[0])
        network = sim_group_size(network,arms,n_time,alg, gamma)[1]
        reciprocity = np.append(reciprocity,reciprocity_weighted_graph(network))
        F = transform_di_weight_simple(network,0.99)
        clique_numbers = np.append(clique_numbers,nx.graph_clique_number(F))
        avg_clustering = np.append(avg_clustering,nx.average_clustering(F))
        #   network.plot_high_sharing(0.1)
        # draw_cliques(F)
    return data, reciprocity, clique_numbers, avg_clustering
    

# get data for for each propensity to share
def get_simulation_data(n_time=1000, n_steps = 10, alg=discounted_thompson, arm_type="bernoulli", gamma=1, comm_init=False) :
    global K
    data = dict()
    reciprocity = dict()
    clique_numbers = dict()
    avg_clustering = dict()
    if arm_type == "bernoulli":
        arm_field = bernoulli_arms(K)
    elif arm_type == "gaussian":
        arm_field = truncated_gaussian_arms(K) 
    for i in range(n_steps+1) :
        #print(str(i/float(n_steps)*100)+'%')
         a,b,c,d = full_sim_group_size(inverse_sigmoid_func(i/float(n_steps)),arm_field, n_time, alg, gamma, comm_init)
         data[i/float(n_steps)*100], reciprocity[i/float(n_steps)*100], clique_numbers[i/float(n_steps)*100], avg_clustering[i/float(n_steps)*100] = a,b,c,d
    return data, reciprocity, clique_numbers, avg_clustering

    
# take multiple simulations and take averages of the obtained data
def simulations(n_iter = 1000, n_time=1000, n_steps = 10, alg=discounted_thompson, arm_type="bernoulli",gamma=1, comm_init = False):
    global K
    data = dict()
    reciprocity = dict()
    clique_numbers = dict()
    avg_clustering = dict()
    for k in range(n_iter):
        print(str(k/float(n_iter)*100)+'%')
        sim_data,recip,cliq,clust = get_simulation_data(n_time, n_steps, alg, arm_type, gamma, comm_init)
        for i in range(n_steps+1):
            data[i/float(n_steps)*100] = (k*data.get(i/float(n_steps)*100,np.zeros(math.ceil((K-1)/5))) + sim_data[i/float(n_steps)*100])/(k+1)
            reciprocity[i/float(n_steps)*100] = (k*reciprocity.get(i/float(n_steps)*100,np.zeros(math.ceil((K-1)/5))) + recip[i/float(n_steps)*100])/(k+1)
            clique_numbers[i/float(n_steps)*100] = (k*clique_numbers.get(i/float(n_steps)*100,np.zeros(math.ceil((K-1)/5))) + cliq[i/float(n_steps)*100])/(k+1)
            avg_clustering[i/float(n_steps)*100] = (k*avg_clustering.get(i/float(n_steps)*100,np.zeros(math.ceil((K-1)/5))) + clust[i/float(n_steps)*100])/(k+1)
    return data, reciprocity, clique_numbers, avg_clustering
    
    
# plots the degree distribution
def plot_results(n_iter = 1000, n_time=1000, n_steps = 20, alg=discounted_thompson, arm_type="bernoulli", plot_title='', gamma = 1, comm_init=False ) :
    global K
    y_text = ['Average pay-off/average oracle pay-off','Reciprocity','Clique number','Average clustering coefficient']
    y_lims = [(0.5,0.9),(0.2,1.05),(0,5),(-0.05,1.05)]
    for i in range(4):
        sim_data = simulations(n_iter, n_time, n_steps, alg, arm_type, gamma, comm_init)[i]
        #print (sim_data)
        grp_size = [i for i in range(2,K+1,5)]
        plt.figure()
        for key,value in sim_data.items():
            if i>=2:
                if key!=100.0:
                    #print (key,value)
                    plt.plot(grp_size, value, label=key)
            else:
                #print (key,value)
                plt.plot(grp_size, value, label=key)
        plt.xlabel('Group size')
        plt.ylabel(y_text[i])
        plt.title(plot_title)
        plt.ylim(y_lims[i])
        plt.xlim(1,K+30)
        plt.legend(loc="right", title="Propensity to share")
        #plt.show()
        plt.grid(axis='y', alpha=.3)
        # Remove borders
        plt.gca().spines["top"].set_alpha(0.0)    
        plt.gca().spines["bottom"].set_alpha(0.5)
        plt.gca().spines["right"].set_alpha(0.0)    
        plt.gca().spines["left"].set_alpha(0.5)   
        # plt.legend(loc='upper right', ncol=2, fontsize=12)
        plt.show()
            
    
    
def plot_corrs(n_iter = 1000, n_time=1000, n_steps = 20, alg=discounted_thompson, arm_type="bernoulli", plot_title='', gamma = 1, comm_init=False ) :
    global K 
    sim_data = simulations(n_iter, n_time, n_steps, alg, arm_type, gamma, comm_init)[1]
    print (sim_data)
    grp_size = [i for i in range(2,K+1,5)]
    plt.figure()
    for key,value in sim_data.items():
        #print (key,value)
        plt.plot(grp_size, value, label=key)
    plt.xlabel('Group size')
    plt.ylabel('Average in- and out-degree correlation between pairs of nodes')
    plt.title(plot_title)
    plt.ylim(-1,1)
    plt.xlim(1,K+30)
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
def full_sim_run(n_iter = 1000, n_time=1000, n_steps = 20,algs = [discounted_thompson,discounted_thompson_general], arms=["bernoulli","gaussian"], sims_per_alg=1, params = None, gamma=1, comm_init=False) :
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
            plot_results(n_iter, n_time, n_steps, algs[i], arms[i], algs[i].__name__+'/'+arms[i], gamma, comm_init)
    elapsed = int(round(time.time())) - start
    p_message('total time taken: ' + sec_to_string(elapsed))
    #return simulation
    
    
##############################################################################
##############################################################################

K = 50
full_sim_run(n_iter = 50, n_time=20, n_steps = 4)
