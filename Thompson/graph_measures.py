# -*- coding: utf-8 -*-
"""
Network metrics
"""
from collections import defaultdict
import networkx as nx
import numpy as np
from math import *
import matplotlib.pylab as plt
import itertools as it

from communication_network import *


colors=it.cycle('bgrcmyk')# blue, green, red, ...
hatches=it.cycle('/\|-+*')


def mirr_matrix_coeff(A):
    s_x = np.empty(0)
    s_y = np.empty(0)
    for i in range(int(math.sqrt(np.size(A)))):
        for j in range(i+1,int(math.sqrt(np.size(A)))):
            s_x = np.append(s_x,A[i,j])
            s_y = np.append(s_y,A[j,i])
    return np.corrcoef(s_x,s_y)[0,1]
    
    
def adjacency_matrix(network):
    return nx.adjacency_matrix(network).todense()
    

def reciprocity_weighted_graph(network):
    recip_weight = 0
    total_weight = 0
    weighted_matrix = adjacency_matrix(network)
    for i in range(int(sqrt(np.size(weighted_matrix)))):
        for j in range(int(sqrt(np.size(weighted_matrix)))):
            if i!=j:
                recip_weight += min(weighted_matrix[i,j],weighted_matrix[j,i])
                total_weight += weighted_matrix[i,j]
    return recip_weight/total_weight
    
    
def record_data_sharing(network, data_array) :
    """take single iteration snapshot of the mean rewards"""
    for n in network.nodes() :
        data_array[n] = np.sum(network.node[n]['rewards'])
    
        
def record_info_sharing(network, data_array):
    for n in network.nodes():
        data_array[n] = np.size(network.node[n]['information shared with'])
    
        
def record_coop_rates(network, data_array):
    for n in network.nodes():
        data_array[n] = np.mean(network.node[n]['share information'])
    
        
def transform_di_weight_simple(network,treshold):
    G = nx.Graph()
    G.add_nodes_from(network)
    #print (network.edges(data=True))
    for i in range(len(network)):
        for j in range(i):
            if network[i][j]['weight'] > treshold and network[j][i]['weight'] > treshold:
                G.add_edge(i,j)
    return G

def transform_weight_simple(network,treshold):
    G = nx.Graph()
    G.add_nodes_from(network)
    #print (network.edges(data=True))
    for i in range(len(network)):
        for j in range(i):
            if network[i][j]['weight'] > treshold:
                G.add_edge(i,j)
    return G
    
def transform_di_simple(network):
    G = nx.Graph()
    G.add_nodes_from(network)
    for i in range(len(network)):
        for j in range(len(network)):
            if (i,j) in network.edges() and (j,i) in network.edges():
                G.add_edge(i,j)
    return G
    
    
def transform_high_receivers(network,treshold):
    l = {}
    free_riders = []
    for i in range(len(network)):
        for j in range(len(network)):
            if j!=i:
                if network.edge[i][j][0]['weight'] <= treshold and network.edge[j][i][0]['weight'] > treshold:
                    l[i] = l.get(i,0)+1
    for key,value in l.items():
        if value >= 2:
            free_riders.append(key)
    return free_riders
    
    
def high_receivers_simple(network):
    l = {}
    free_riders = []
    for i in range(len(network)):
        for j in range(len(network)):
            if j!=i:
                if (i,j) in network.edges() and (j,i) in network.edges():
                    l[i] = l.get(i,0)+1
    for key,value in l.items():
        if value >= 2:
            free_riders.append(key)
    return free_riders


def k_clique_communities(G, k, cliques=None):
    """
    Adapted from the networkx library. 
    
    Find k-clique communities in graph using the percolation method.

    A k-clique community is the union of all cliques of size k that
    can be reached through adjacent (sharing k-1 nodes) k-cliques.

    Parameters:
    G : NetworkX graph (undirected)

    k : int
       Size of smallest clique

    cliques: list or generator       
       Precomputed cliques (use networkx.find_cliques(G))

    References:
    .. [1] Gergely Palla, Imre Derényi, Illés Farkas1, and Tamás Vicsek,
       Uncovering the overlapping community structure of complex networks 
       in nature and society Nature 435, 814-818, 2005,
       doi:10.1038/nature03607
    """
    if k < 2:
        raise nx.NetworkXError("k=%d, k must be greater than 1." % k)
    if cliques is None:
        cliques = nx.find_cliques(G)
    cliques = [frozenset(c) for c in cliques if len(c) >= k]

    # First index which nodes are in which cliques
    membership_dict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            membership_dict[node].append(clique)

    # For each clique, see which adjacent cliques percolate
    perc_graph = nx.Graph()
    perc_graph.add_nodes_from(cliques)
    for clique in cliques:
        for adj_clique in _get_adjacent_cliques(clique, membership_dict):
            if len(clique.intersection(adj_clique)) >= (k - 1):
                perc_graph.add_edge(clique, adj_clique)

    # Connected components of clique graph with perc edges
    # are the percolated cliques
    for component in nx.connected_components(perc_graph):
        yield(frozenset.union(*component))    


def _get_adjacent_cliques(clique, membership_dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques


def detect_k_communities(network,treshold,k,cliques=None):
    G = transform_di_weight_simple(network,treshold)
    return [list(x) for x in k_clique_communities(G, k, cliques)]


def detect_k_communities_undirected(network,treshold,k,cliques=None):
    G = nx.Graph()
    G.add_nodes_from(network)
    for i in range(len(network)):
        for j in range(i):
            if network[i][j]['weight'] <= treshold:
                G.add_edge(i,j)
    return [list(x) for x in k_clique_communities(G, k, cliques)]

    
def draw_circle_around_clique(clique,coords):
    dist=0
    temp_dist=0
    center=[0 for i in range(2)]
    color=next(colors)
    for a in clique:
        for b in clique:
            temp_dist=(coords[a][0]-coords[b][0])**2+(coords[a][1]-coords[b][1])**2
            if temp_dist>dist:
                dist=temp_dist
                for i in range(2):
                    center[i]=(coords[a][i]+coords[b][i])/2
    rad=dist**0.5/2
    cir = plt.Circle((center[0],center[1]),   radius=rad*1.3,fill=False,color=color,hatch=next(hatches))
    plt.gca().add_patch(cir)
    plt.axis('scaled')
    # return color of the circle, to use it as the color for vertices of the cliques
    return color

    
def draw_cliques(G):
    global colors, hatches
    
    # remember the coordinates of the vertices
    coords=nx.spring_layout(G)
    
    # remove "len(clique)>2" if you're interested in maxcliques with 2 edges
    cliques=[clique for clique in nx.find_cliques(G) if len(clique)>2]
    
    #draw the graph
    nx.draw(G,pos=coords)
    for clique in cliques:
        nx.draw_networkx_nodes(G,pos=coords,nodelist=clique,node_color=draw_circle_around_clique(clique,coords))
    
    plt.show()
    

def overall_reciprocity(G):
    """Compute the reciprocity for the whole graph.
    """
    n_all_edge = G.number_of_edges()
    n_overlap_edge = (n_all_edge - G.to_undirected().number_of_edges()) * 2

    if n_all_edge == 0:
        raise NetworkXError("Not defined for empty graphs")

    return float(n_overlap_edge) / float(n_all_edge)
    
    
"""
G=nx.gnp_random_graph(n=20,p=0.5)
draw_cliques(G)
print([list(x) for x in nx.k_clique_communities(G,4)])
print(nx.graph_clique_number(G))
"""