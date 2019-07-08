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

plt.style.use('seaborn-white')
np.random.seed(None)

from communication_network import *
from mab_algorithms import *
from arms import *
from graph_measures import *

global K
K = 50
    
            
##############################################################################
"""
Create simulations for various group sizes
"""
##############################################################################

    


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
"""
K = 50
full_sim_run(n_iter = 50, n_time=20, n_steps = 4)
"""

##############################################################################
"""
Create time series with a fixed group size
"""
##############################################################################

class Simulation:
    
    #### initialise
    
    def __init__(self, n_agents, n_bandits, type_bandits, type_alg, const=True, prop_share = 0):
        """
        Bandits have several options, give one of those options as type_bandits:
            1. bernoulli_arms
            2. truncated_gaussian_arms
        Type of algorithms:
            1. discounted_thompson
            2. discounted_thompson_general
        """

        self.n_agents = n_agents
        self.n_bandits = n_bandits
        self.type_bandits = type_bandits
        self.type_alg = type_alg
        self.agents = self.__generate_agents(const,prop_share)
        self.bandits = self.__generate_bandits()
        self.oracle = self.__find_oracle()

        
    def __generate_agents(self,const=True,prop_share=0):
        G = nx.MultiDiGraph()
        A = comm_graph(G)
        A.initialise(self.n_agents, self.n_bandits, self.type_alg)
        A.init_info_further(self.n_bandits)
        A.init_sharing_vector(const, prop_share)
        A.update_sharing_weights()
        return A
        
        
    def __generate_bandits(self):
        return self.type_bandits(self.n_bandits)
        
    
    def __find_oracle(self):
        return np.argmax(np.array(self.bandits.field_exp))
            
        
    def __generate_lattice(self):
        """
        Default Lattice has only 4 adges(vertical&horizontal), so adding 4 edges in diagonal direction and 
        Set periodic boundary condition (toroidal surface)
        """

        n = math.sqrt(self.n_bandits)    # n×n lattice is generated
        G = nx.grid_graph(dim = [n,n]) 

        # Add diagonal edge except for outer edge agent
        for i in range(1,n-1):
            for j in range(1,n-1):
                G.add_edge((i,j), (i+1,j+1))
                G.add_edge((i,j), (i+1,j-1))
                G.add_edge((i,j), (i-1,j+1))
                G.add_edge((i,j), (i-1,j-1))
            
        # Add edge along i = 0, j=1~n-2
        for j in range(1,n-1):
            G.add_edge((0,j), (n-1,j))
            G.add_edge((0,j), (n-1,j+1))
            G.add_edge((0,j), (n-1,j-1))
            G.add_edge((0,j), (1,j-1))
            G.add_edge((0,j), (1,j+1))
        
        # Add edge along j=0, i=1~n-2
        for i in range(1,n-1): 
            G.add_edge((i,0), (i,n-1))
            G.add_edge((i,0), (i-1,n-1))
            G.add_edge((i,0), (i+1,n-1))
            G.add_edge((i,0), (i+1,1))
    
        # Add edge along j=0
        G.add_edge((0,0), (n-1,0))
        G.add_edge((0,0), (n-1,0+1))
        G.add_edge((0,0), (n-1,n-1))
        G.add_edge((0,0), (0,n-1))
        G.add_edge((0,0), (1,n-1))
  
        # Add edge along j=n-1
        G.add_edge((0,n-1), (n-1,n-1))
        G.add_edge((0,n-1), (n-1,0))
        G.add_edge((0,n-1), (n-1,n-2))
        G.add_edge((0,n-1), (0,0))
    
        # Add edge along i=n-1
        G.add_edge((n-1,0), (0,0))
        G.add_edge((n-1,0), (0,1))
        G.add_edge((n-1,0), (0,n-1))
        G.add_edge((n-1,0), (n-1,n-1))
        G.add_edge((n-1,0), (n-2,n-1))
           
        # Upper right edge agent
        G.add_edge((n-1,n-2),(n-2,n-1))
        
        return G

        
    def choose_initial_cooperators(self):
        population = self.n_agents
        self.initial_cooperators = rnd.sample(range(population), k = int(population/2))

        
    def initialize_strategy(self):
        """Initialize the strategy of agents (IN CASE WE IMPLEMENT TIT-FOR-TAT)"""
        for n in range(self.n_agents):
            if n in self.initial_cooperators:
                self.agents.node[n]['strategy'] = "C"
            else:
                self.agents.node[n]['strategy'] = "D"

                
    def count_payoff(self):
        """Count the payoff based on payoff matrix or a continous function
        in case of simulating adaptive dynamics"""
        pass

                    
    def update_strategy(self):
        pass

            
    #### Measures
    
    def count_fc(self):
        """Calculate the fraction of cooperative agents"""
        fc = len([agent for agent in self.agents if agent.strategy == "C"])/len(self.agents)
        return fc
        
        
    def summarise_weights(self):
        """Calculate the average propensity to share information"""
        all_weights = np.array([self.agents.edges(data=True)[i][2]['weight'] for i in range(len(self.agents.edges()))])
        #print (all_weights)
        return np.mean(all_weights), np.var(all_weights)
        
        
    def reciprocity(self):
        """Calculate the reciprocity of the network"""
        return reciprocity_weighted_graph(self.agents)
        
    
    def clique_n(self):
        """Calculate the clique number of the network"""
        F = transform_di_weight_simple(self.agents,0.95)
        return nx.graph_clique_number(F)
        
        
    def clustering_coef(self):
        """Calculates the average clustering coefficient of the network"""
        F = transform_di_weight_simple(self.agents,0.95)
        return nx.average_clustering(F)
        
        
    def degree_assortativity(self):
        """Calculates the degree assortativity of the graph"""
        F = transform_di_weight_simple(self.agents,0.95)
        return nx.degree_assortativity_coefficient(F)
        
    
    #### The game
        
    def one_episode(self,t=None):
        """Play and instance of the game - agents choose an arm and share information"""        
        self.agents.select_arm(self.bandits, self.type_alg)
        self.agents.share_info()
        self.agents.evolve_sharing()
        self.agents.update_sharing_weights()
        
        
    def avg_pay_off(self,t=None):
        F = transform_di_weight_simple(self.agents,0.95)
        max_clique_membership = [nx.node_clique_number(F,i)>=3 for i in F.nodes()]
        #print (max_clique_membership)
        in_clique = np.array([self.agents.node[i]['rewards'][-1] for i in [value for value in range(len(max_clique_membership)) if max_clique_membership[value]==True]])
        not_in_clique = np.array([self.agents.node[i]['rewards'][-1] for i in [value for value in range(len(max_clique_membership)) if max_clique_membership[value]==False]])
        prop_clique = np.size(in_clique)/self.n_agents
        #print (in_clique,not_in_clique,np.array([self.agents.node[i]['rewards'][-1] for i in range(len(self.agents))]))
        mean_clique_members = np.mean(in_clique)
        mean_non_clique_members = np.mean(not_in_clique)
        mean_all = np.mean(np.array([self.agents.node[i]['rewards'][-1] for i in range(len(self.agents))]))
        oracle_performance = self.bandits.field[self.oracle].draw_sample()
        return mean_all, mean_clique_members, mean_non_clique_members, prop_clique
        
        
    def play_game(self):
        t_max = 500
        
        data = dict()
        
        for t in range(1,t_max+1):
            self.one_episode()
            mean_all,mean_clique,mean_non_clique, prop_clique = self.avg_pay_off()
            avg_weight,weight_var = self.summarise_weights()
            reciprocity = self.reciprocity()
            #print (reciprocity)
            clique_number = self.clique_n()
            clustering_coef = self.clustering_coef()
            #assortativity = self.degree_assortativity()
            data[t] = mean_all,mean_clique,mean_non_clique,reciprocity,clique_number,clustering_coef, avg_weight,weight_var, prop_clique

        return data
        
    
    def plot_data(self):
        t_max = 500
        y_text = ['Average pay-off','Average pay-off(C)','Average pay-off(NC)','Reciprocity','Clique number','Average clustering coefficient','Average propensity to share','Variance of propensity to share','Proportion of agents in cliques']
        y_lims = [(0,1.05),(0,1.05),(0,1.05),(0.2,1.05),(0,5),(-0.05,1.05),(0,1),(0,1),(0,1)]
        data = self.play_game()
        for i in range(len(y_lims)):
            sim_data = [data[t][i] for t in range(1,t_max+1)]
            #print (sim_data)
            time_axis = [j for j in range(1,t_max+1)]
            plt.figure()
            plt.plot(time_axis, sim_data)
            plt.xlabel('Time')
            plt.ylabel(y_text[i])
            plt.title('')
            plt.ylim(y_lims[i])
            plt.xlim(0,101)
            #plt.legend(loc="right", title="Propensity to share")
            #plt.show()
            plt.grid(axis='y', alpha=.3)
            # Remove borders
            plt.gca().spines["top"].set_alpha(0.0)    
            plt.gca().spines["bottom"].set_alpha(0.5)
            plt.gca().spines["right"].set_alpha(0.0)    
            plt.gca().spines["left"].set_alpha(0.5)   
            # plt.legend(loc='upper right', ncol=2, fontsize=12)
            plt.show()
        
            
class Multi_Sim:
    
    def __init__(self, n_agents, n_bandits, type_bandits, type_alg, n_steps):
        self.n_agents = n_agents
        self.n_bandits = n_bandits
        self.type_bandits = type_bandits
        self.type_alg = type_alg
        self.n_steps = n_steps
        self.sims = self.__generate_sims()
        
    def __generate_sims(self):
        sims = []
        for i in range(self.n_steps+1):
            sims.append(Simulation(self.n_agents, self.n_bandits, self.type_bandits, self.type_alg, True, i/self.n_steps))
        return sims
        
    def get_data(self):
        data = dict()
        for i in range(self.n_steps+1):
            data[i/self.n_steps] = self.sims[i].play_game()
        return data
        
    def plot_data(self):
        y_text = ['Average pay-off','Average pay-off(C)','Average pay-off(NC)','Reciprocity','Clique number','Average clustering coefficient','Average propensity to share','Variance of propensity to share','Proportion of agents in cliques']
        y_lims = [(0,1.05),(0,1.05),(0,1.05),(0.2,1.05),(0,5),(-0.05,1.05),(0,1),(0,1),(0,1)]
        data = self.get_data()
        time_axis = [j for j in range(1,101)]
        for i in range(len(y_lims)):
            plt.figure()
            for j in range(self.n_steps+1):
                sim_data = [data[j/self.n_steps][t][i] for t in range(1,101)]
                #print (sim_data)
                plt.plot(time_axis, sim_data, label = round(j/self.n_steps,2))
            plt.xlabel('Time')
            plt.ylabel(y_text[i])
            plt.title('')
            plt.ylim(y_lims[i])
            plt.xlim(0,130)
            plt.legend(loc='best', title="Propensity to share")
            #plt.show()
            plt.grid(axis='y', alpha=.3)
            # Remove borders
            plt.gca().spines["top"].set_alpha(0.0)    
            plt.gca().spines["bottom"].set_alpha(0.5)
            plt.gca().spines["right"].set_alpha(0.0)    
            plt.gca().spines["left"].set_alpha(0.5)   
            # plt.legend(loc='upper right', ncol=2, fontsize=12)
            plt.show()
    


def main():
    """
    S = Simulation(10, 20, bernoulli_arms, discounted_thompson, const=False)
    for i in range(20):
        S.one_episode()
    print (S.agents.nodes(data=True))
    """
    #D = Simulation(10, 20, bernoulli_arms, discounted_thompson, True)
    #print (D.agents.nodes(data=True))
    #print (S.agents.nodes(data=True))
    MS = Multi_Sim(20, 50, bernoulli_arms, discounted_thompson, 3)
    MS.plot_data()
