# -*- coding: utf-8 -*-
"""
Network of interactions
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
import copy

from mab_algorithms import *
from arms import *


# K is number of arms of the multi-armed bandit
global b, N, payoff, K

K = 100


def add_weights_complete_graph(graph, comm_prop = 0.5, const = True):
    """
    Add weights to a graph.
    
    Inputs:
    graph - networkx type graph
    const - True if the weights are constant
    comm_prop - communication propensity in the network
    
    Output:
    weighted graph
    """
    
    return graph.add_weighted_edges_from((u,v,comm_prop) for u,v in graph.edges())



def comm_graph_heavyside(n,a):
    """
    Produces a complete graph with weights according to a smooth approximation to a heavy-side step function
    
    Inputs:
    n - number of nodes
    a - parameter input to the heavy-side step function
    
    Output:
    complete weighted graph
    """
    
    # intialise the graph
    G=nx.complete_graph(n)
    
    # add edges with weights
    G.add_weighted_edges_from((u,v,heavy_side_step(u,v,a)) for u,v in G.edges())
    
    return G


def sharing_line(x,y,a):
    """
    Compute a smooth approximation to a heavy-side step function of distance between x,y using the usual Euclidean metric
    """
    return 1/(1+math.exp(a*abs(x-y)))

    
def sharing_cycle(x,y,a):
    """
    Computes the propensity to share information between two nodes based on the length of the shortest path on a cycle.
    """
    global N
    G = nx.cycle_graph(N)
    d = dijkstra_path(G, x, y)
    return 1/(1+math.exp(a*d))

    
def initialise_network_arms(n_players, arms,alg=discounted_thompson,gamma = 1,comm_init=False,field=[]):
    """
    Initialises a position of individuals with regards to arms of the MAB
    """
    network = nx.complete_graph(n_players)
    for i in range(n_players):
        network.node[i]['alloc_seq'] = np.empty(0) #this will be the array of payoff from the multi-armed bandit
        network.node[i]['rewards'] = np.empty(0)
        network.node[i]['S'] = np.zeros(n_players) # for Thompson algs
        network.node[i]['F'] = np.zeros(n_players) # for Thompson algs
    arm_selection(network,arms, alg, gamma, field)
    return network
    
    
# initialise payoff and strategy fields
def init(network) :
    """adds information to each node
    """
    #global N 
    n = len(network)
    cnt = 0
    network.graph['degrees'] = [network.degree(node) for node in network.nodes()] #just to check for now, as we know the graph is complete
    for i in range(n) :
        network.node[i]['payoff'] = 0 #average pay-off from all observed rewards
        network.node[i]['strategy'] = None #UPDATE
        network.node[i]['next strategy'] = None
        network.node[i]['neighbors'] = network.neighbors(i) #all other nodes for now
        network.node[i]['fitness'] = 0
        network.node[i]['arm'] = 0 #current location of the agent
        network.node[i]['prior means'] = np.zeros(K)
        network.node[i]['m_bar'] = np.zeros(K)
        network.node[i]['n'] = np.zeros(K)
        network.node[i]['prior variance'] = 1
        network.node[i]['variance'] = 1
        network.node[i]['game played with'] = np.empty(0) #with whom was the game already played? i.e. with whome has this individual tried
                                                        # to share information?
        network.node[i]['information shared with'] = np.empty(0) # with whom of the neighbors was a subset of the information set during
                                                            # the particular iteration?

        
    # TO-DO fix later with evolutionary strategies - distribute evenly and randomly among players
    """while cnt < n/2 :
        s = mt.floor(rd.random()*n)
        if not network.node[s]['strategy'] :
            network.node[s]['strategy'] = True
            cnt = cnt + 1"""
            
            
            
def initialise_pos(network,field):
    """
    Initialise the position of the agents in the field
    """
    for i in range(len(network)):
        if np.all(network.node[i]['prior means']==network.node[i]['prior means'][0]):
            arm = random.randint(0,K)
            network.node[i]['arm'] = arm
            network.node[i]['n'][arm] += 1
            network.node[i]['rewards'][arm] = field_to_arms(field)[arm]




def arm_selection(network,arms,alg=discounted_thompson,gamma = 1, field=[],first_iter=False):
    """
    Performs one round when the agents select an arm based on a specified algorithm and assign the allocation
    sequence and array of rewards to the corresponding agent.
    
    Inputs:
    network - network of social interactions
    field - Gaussian random field
    alg - which allocation algorithm will be used (default: deterministic)
    """
    global K
    
    if alg == deterministic_ucl or alg == block_ucl:
        arms = field_to_arms(field)
        for i in range(len(network)):
            priors = network.node[i]['prior mean']
            prior_var = network.node[i]['prior variance']
            variance = network.node[i]['variance']
            n_arms = np.len(arms)
            n_visits = network.node[i]['n']
            avg_arms = network.node[i]['m_bar']
            time_horizon = 1
            a,b,n,m_bar = alg(priors,prior_var,variance,n_arms,time_horizon,arms,n_visits, avg_arms,random_rewards=False)
            network.node[i]['alloc_seq'] = np.append(network.node[i]['alloc_seq'],field_to_arms(a,math.sqrt(n_arms)))
            network.node[i]['rewards'] = np.append(network.node[i]['rewards'],b)
            network.node[i]['prior mean'] = np.mean(network.node[i]['rewards']) #update priors to have them ready for next iteration 
            network.node[i]['n'] = network.node[i]['n'] + n
            network.node[i]['m_bar'] = network.node[i]['m_bar'] + m_bar
    

    elif alg == discounted_thompson or alg == discounted_thompson_general:
        for i in range(len(network)):
            alpha_0 = beta_0 = 1
            S = network.node[i]['S']
            F = network.node[i]['F']
            T = 1
            alloc_seq,reward_seq,S,F = alg(arms,gamma,alpha_0,beta_0,K, T, S,F, first_iter)
            network.node[i]['alloc_seq'] = np.append(network.node[i]['alloc_seq'],alloc_seq)
            network.node[i]['rewards'] = np.append(network.node[i]['rewards'],reward_seq)
            network.node[i]['S'] = network.node[i]['S'] + S
            network.node[i]['F'] = network.node[i]['F'] + F
    else:
        print ('ERROR, a mistake in this function')
        


def information_sharing_thompson(network):
    """
    Performs one step of information sharing
    """
    for i in range(len(network)):
        network.node[i]['game played with'] = np.empty(0)
        network.node[i]['information shared with'] = np.empty(0)
    for i in range(len(network)):
        to_iterate_over = np.setdiff1d(network.node[i]['neighbors'],network.node[i]['game played with'])
        for j in to_iterate_over:
            network.node[i]['game played with'] = np.append(network.node[i]['game played with'],j)
            network.node[j]['game played with'] = np.append(network.node[j]['game played with'],i)
            rand_n = random.random()
            if network.edge[i][j]['weight'] > rand_n:
                network.node[j]['information shared with'] = np.append(network.node[j]['information shared with'],i)
                network.node[i]['information shared with'] = np.append(network.node[i]['information shared with'],j)
                # i shares with j:
                s = network.node[i]['alloc_seq'][-1]
                if network.node[i]['rewards'][-1]==1:
                    network.node[j]['S'][s] += 1
                else:
                    network.node[j]['F'][s] += 1
                # j shares with i:
                s = network.node[j]['alloc_seq'][-1]
                if network.node[j]['rewards'][-1]==1:
                    network.node[i]['S'][s] += 1
                else:
                    network.node[i]['F'][s] += 1
                

def community_init(network) :
    """Initialise pay-offs based on community membership"""
    n = len(network)
    partition = community.best_partition(network)
    community_index = sorted(set(partition.values()))
    num_communities = max(community_index)+1
    comm_strat = [True if k < num_communities/2 else False for k in community_index]
    rd.shuffle(comm_strat)
    network.graph['degrees'] =  [network.degree(node) 
                                for node in network.nodes()]
    for i in range(n) :
        network.node[i]['strategy'] = comm_strat[partition[i]]
        network.node[i]['next strategy'] = None
        network.node[i]['neighbors'] = network.neighbors(i)
        network.node[i]['payoff'] = 0 #average pay-off from all observed rewards
        network.node[i]['alloc_seq'] = [] #this will be the array of payoff from the multi-armed bandit
        network.node[i]['rewards'] = []
        network.node[i]['fitness'] = 0
        network.node[i]['arm'] = 0 #current location of the agent

"""
arms = bernoulli_arms(K)
for i in range(K):
    print (arms.field[i].p)
G = initialise_network_arms(K, arms)
print (G.nodes(data=True))

H = copy.deepcopy(G)
net = H.subgraph([j for j in range(4)])
init(net)
add_weights_complete_graph(net, 0.5, const = True)
information_sharing_thompson(net)

print (net.nodes(data=True))
"""