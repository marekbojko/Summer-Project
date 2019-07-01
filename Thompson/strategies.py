# -*- coding: utf-8 -*-
"""
EGT strategies
"""

import random as rd

global b, N, payoff, K

def record_data(network, data_array):
    """take single iteration snapshot of node strategies"""
    for n in network.nodes() :
        data_array[n] = network.node[n]['strategy']



def new_strategy(network, node):
    """return new strategy for a single node"""
    global b
    if not network.node[node]['neighbors'] : #in the case of an isolated vertex
        return network.node[node]['strategy']
    v = rd.choice(network.node[node]['neighbors'])
    if network.node[node]['payoff'] < network.node[v]['payoff'] :
        p = (network.node[v]['payoff'] - network.node[node]['payoff'])/(max(network.graph['degrees'][node], network.graph['degrees'][v])*b*100.0)
        s = rd.random()
        if (s < p) :
            return network.node[v]['strategy']
    return network.node[node]['strategy']


def strategy_update(network):
    """update strategy of each node"""
    new_strategies = [new_strategy(network, u) for u in network.nodes()]
    for q in network.nodes() :
        network.node[q]['strategy'] = new_strategies[q]

