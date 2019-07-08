# -*- coding: utf-8 -*-
"""
Network of interactions
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
import copy

from mab_algorithms import *
from arms import *



# K is number of arms of the multi-armed bandit
global b, N, payoff, K

K = 50


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

    
def sharing_cycle(x,y,n_agents):
    """
    Computes the propensity to share information between two nodes based on the length of the shortest path on a cycle.
    """
    G = nx.cycle_graph(n_agents)
    d = len(nx.shortest_path(G, x, y))
    return -d/10
    
    
# return the acumulated payoff of all nodes
def accum_payoff(network) :
    payoff = np.zeros(len(network))
    for i in range(len(network)):
        payoff[i] = np.sum(network.node[i]['rewards'])
    return payoff
    

def cooling_schedule(t):
    return 1/math.log(t)
    
    
def accum_payoff_exp(network,t):
    payoff = np.zeros(len(network))
    for i in range(len(network)):
        payoff[i] = math.exp(np.sum(network.node[i]['rewards'])/cooling_schedule(t))
    return payoff
    
    
def sigmoid_func(x):
    try:
        ans = 1 / (1 + math.exp(-x))
    except OverflowError:
        if x<-1000:
            ans = 0
        else:
            ans = 1
    return ans
    
    
def inverse_sigmoid_func(x):
    try:
        ans = math.log(x/(1-x))
    except ZeroDivisionError:
        ans = math.inf
    except ValueError:
        ans = -math.inf
    return ans
    

def sigmoid(d):
    b=dict()
    for key,value in d.items():
        b[key] = sigmoid_func(value)
    return b
    
    
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
    

def init_sharing(network,const = True, prop_share = 0, func = sharing_cycle):
    for i in range(len(network)):
        neighbors = network.node[i]['neighbors']
        if const:
            sharing_vector = [inverse_sigmoid_func(prop_share) for j in range(len(network.neighbors(i)))]
        else:
            sharing_vector = [sharing_cycle(i,j,len(network)) for j in neighbors] # TO-DO: UPDATE
        network.node[i]['propensity to share'] = {neighbors[i]: sharing_vector[i] for i in range(len(sharing_vector))}
        network.node[i]['share information'] = sigmoid(network.node[i]['propensity to share'])

def add_noise_dict(d,mu,sigma):
    b = dict()
    for key,value in d.items():
        if sigma!=0:
            b[key] = value + np.random.normal(mu,sigma)
        else:
            b[key] = value
    return b
    
    
def sim_annealing_cooling(network,t):
    mean_accum_payoff = np.mean(accum_payoff_exp(network,t))
    for i in range(len(network)):
        acc_payoff = exp(np.sum(network.node[i]['rewards'])/cooling_schedule(t))
        epsilon = mean_accum_payoff/acc_payoff
        network.node[i]['propensity to share'] = add_noise_dict(network.node[i]['propensity to share'],0,epsilon)
        network.node[i]['share information'] = sigmoid(network.node[i]['propensity to share'])
        
    
def evolve_sharing(network):
    mean_accum_payoff = np.mean(accum_payoff(network))
    for i in range(len(network)):
        acc_payoff = np.sum(network.node[i]['rewards'])
        if acc_payoff == 0:
            if mean_accum_payoff == 0:
                epsilon = 0
            else:
                epsilon = 1
        else:
            epsilon = mean_accum_payoff/acc_payoff
        network.node[i]['propensity to share'] = add_noise_dict(network.node[i]['propensity to share'],0,epsilon)
        network.node[i]['share information'] = sigmoid(network.node[i]['propensity to share'])
    
        
# initialise payoff and strategy fields
def init(network,K) :
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
            network.node[i]['S'] = S
            network.node[i]['F'] = F
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


class comm_graph(nx.MultiDiGraph):
    """Some useful methods for multi directed graph"""
    pos=None
    _label=None
    
    
    def add_nodes(self,n):
        self.add_nodes_from([i for i in range(n)])
    
    
    def __str__(self):
        if self._label:
            return self._label
        return nx.MultiDiGraph.__repr__(self)
    
    
    def add_e(self, complete = True, prop_share = 0, e = []):
        if complete:
            self.add_weighted_edges_from((u,v,prop_share) for u in range(len(self)) for v in range(len(self)) if u!=v) 
        else:
            self.add_weighted_edges_from(e)
            
            
    def init_info_main(self, n_arms, alg=discounted_thompson, gamma = 1, comm_init=False, field=[]):
        """
        Initialises a position of individuals with regards to arms of the MAB
        """
        for i in range(len(self)):
            self.node[i]['alloc_seq'] = np.empty(0) #this will be the array of payoff from the multi-armed bandit
            self.node[i]['rewards'] = np.empty(0)
            self.node[i]['S'] = np.zeros(n_arms) # for Thompson algs
            self.node[i]['F'] = np.zeros(n_arms) # for Thompson algs

    
    def init_info_further(self,n_arms):
        init(self,n_arms)
        
    
    def init_sharing_vector(self,const = True, prop_share = 0, func = sharing_cycle):
        init_sharing(self,const, prop_share, func)
        
    
    def initialise(self,n,n_arms,alg=discounted_thompson, gamma = 1,complete=True,const = True,prop_share = 0, comm_init=False, e = [], func=None, field=[]):
        self.add_nodes(n)
        self.add_e(complete,prop_share,e)
        self.init_info_main(n_arms,alg,gamma,comm_init)
    
    
    def select_arm(self,arms,alg=discounted_thompson,gamma = 1, field=[],first_iter=False):
        arm_selection(self,arms,alg,gamma, field,first_iter)
        
    
    def share_info(self):
        """
        Performs one step of information sharing
        """
        for i in range(len(self)):
            self.node[i]['information shared with'] = np.empty(0)
        for i in range(len(self)):
            for j in self.node[i]['neighbors']:
                #print (i,j)
                rand_n = random.random()
                #print (self.edge[i][j][0]['weight'])
                if self.edge[i][j][0]['weight'] > rand_n:
                    self.node[i]['information shared with'] = np.append(self.node[i]['information shared with'],j)
                    # i shares with j:
                    s = self.node[i]['alloc_seq'][-1]
                    if self.node[i]['rewards'][-1]==1:
                        self.node[j]['S'][s] += 1
                    else:
                        self.node[j]['F'][s] += 1

        
    def evolve_sharing(self):
        evolve_sharing(self)
        
    
    def sim_annealing(self,t):
        sim_annealing_cooling(self,t)
        
    
    def update_sharing_weights(self):
        for i in range(len(self)):
            #print (self.nodes(data=True))
            for j in self.node[i]['neighbors']:
                self.edge[i][j][0]['weight'] = self.node[i]['share information'][j]
    
    
    def run_round_sim(self,arms,alg=discounted_thompson,gamma = 1, field=[],first_iter=False):
        self.share_info()
        self.evolve_sharing()
        self.update_sharing_weights()
        self.select_arm(arms, alg ,gamma, field, first_iter)
        
    
        
    def levels_layout(self,ranks=None,xpos=None):
        pos={}
        refpos=nx.spring_layout(self)
        if not isinstance(xpos,dict):
            xpos={i:j for i,j in zip(self.nodes(),xpos ) }
        maxx,maxy=np.max(refpos.values(),axis=0)
        for i,d in self.nodes(data=True):
            if not xpos is None and i in xpos:
                x=xpos[i]
            else:
                x= np.random.random()*maxx
            if ranks is None:
                y=d['height']*maxy
            else:
                y=ranks[i]
            pos[i]=x,y
        self.pos=pos

        
    def plot(self,newfig=True,hold=False,labels=None,edge_labels=None,nscale=1,minsize=0.001,**kwargs):
        '''Use matplotlib to plot the self'''
        
        if self.pos is None:
            pos=self.pos=nx.spring_layout(self)
        else:
            pos=self.pos
        if newfig:
            plt.figure()
        node_size=np.ones(len(self.node) ) *kwargs.get('node_size',1)
        node_size[node_size<minsize]=0
        node_size/=np.max(node_size)/2.
        node_size[node_size>0]=np.clip(node_size[node_size>0],0.1,None)*nscale


        node_size[np.random.random(node_size.shape )<kwargs.get('subsample',0)  ]=0
        nodidx={n:i for i,n in enumerate(self.nodes()) }

        edge_width=kwargs.get('edge_width',1)
        if edge_width:
            sign=kwargs.get('sign',0)
            if not sign or sign<0:
                nx.draw_networkx_edges(self,pos=pos,edgelist=[(i,j) for i,j,d in
                    self.edges(data=True) if d.get('weight',0)<0 and node_size[nodidx[i]]*node_size[nodidx[j]]>0],
                    edge_color='r',width=edge_width )
            if not sign or sign>0:
                nx.draw_networkx_edges(self,pos=pos,edgelist=[(i,j) for i,j,d in
                    self.edges(data=True) if d.get('weight',0)>=0 and node_size[nodidx[i]]*node_size[nodidx[j]]>0.],
                    edge_color='b',width=edge_width )
        nx.draw_networkx_nodes(self,pos=pos ,node_size=node_size*200.,linewidths=0,
                node_color= 'blue')
        if labels is None:
            labels={n:str(n) for n in self.nodes() }
        elif not isinstance(labels,dict):
            labels={n:labels[i] for i,n in enumerate(self.nodes()) }
        nx.draw_networkx_labels(self,pos=pos,labels={n:l for n,l in labels.items() if node_size[nodidx[n]]>0} )
        if not edge_labels is None:
            if edge_labels=='weights':
                #print [self.edge[j][i][0] for i,j,d in
                    #self.edges(data=True)]
                edge_labels={(i,j):'{:.2f},{:.2f}'.format(self.edge[i][j][0]['weight'],self.edge[j][i][0]['weight']) for i,j,d in
                    self.edges(data=True) if # self.edge[i][j][0]['weight']>-self.edge[j][i][0]['weight'] and
                      node_size[nodidx[i]]*node_size[nodidx[j]]>0}
            nx.draw_networkx_edge_labels(self,pos=pos,edge_labels= edge_labels )
        if 'title' in kwargs:
            plt.title(kwargs['title'])
        if 'xlabel' in kwargs:
            plt.xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            plt.ylabel(kwargs['ylabel'])
        plt.axis('off')
        if not hold:
            plt.show()
            
            
    def plot_high_sharing(self,err):
        elarge = [(u, v) for (u, v, d) in self.edges(data=True) if d['weight'] > 1-err]
        
        pos = nx.spring_layout(self)  # positions for all nodes
        
        # nodes
        nx.draw_networkx_nodes(self, pos, node_size=700)
        
        # edges
        nx.draw_networkx_edges(self, pos, edgelist=elarge, width=1)
        
        # labels
        nx.draw_networkx_labels(self, pos, font_size=20, font_family='sans-serif')
        
        plt.axis('off')
        plt.show()
        
        
        
    def plot_high_low(self,err):
        elarge = [(u, v) for (u, v, d) in self.edges(data=True) if d['weight'] > 1-err]
        esmall = [(u, v) for (u, v, d) in self.edges(data=True) if d['weight'] <= err]
        
        pos = nx.spring_layout(self)  # positions for all nodes
        
        # nodes
        nx.draw_networkx_nodes(self, pos, node_size=700)
        
        # edges
        nx.draw_networkx_edges(self, pos, edgelist=elarge, width=1)
        nx.draw_networkx_edges(self, pos, edgelist=esmall, width=1, alpha=0.5, edge_color='b', style='dashed')
        
        # labels
        nx.draw_networkx_labels(self, pos, font_size=20, font_family='sans-serif')
        
        plt.axis('off')
        plt.show()
        
        
