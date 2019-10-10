# -*- coding: utf-8 -*-
"""
Multi-armed bandits algorithms
"""


import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
import math
from scipy import stats
import pickle
import community
import scipy
import scipy.fftpack
import sys
from random import gauss
from numpy.random import standard_normal
from scipy.special import erf
from scipy.stats import norm
from scipy.stats import bernoulli
import scipy.fftpack


#sns.set_style('darkgrid')
np.random.seed(None)
#
random.seed()

from arms import *


def find_min_diff(arr, n): 
    """
    Returns minimum difference between any pair of elements in an array
    
    arr - numpy array
    n - number of elements in the array used in the process
    """
    
    # Sort array in non-decreasing order 
    arr = sorted(arr) 
  
    # Initialize difference as infinite 
    diff = 10**20
  
    # Find the min diff by comparing adjacent 
    # pairs in sorted array 
    for i in range(n-1): 
        if abs(arr[i+1] - arr[i]) < diff: 
            diff = abs(arr[i+1] - arr[i]) 
  
    # Return min diff 
    return diff
    
    
def stochastic_ucl(prior_means,prior_variance,variance,n_arms,time_horizon,arms):
    """
    Inputs: 
    prior_means - an array of prior means about the pay-offs of the arms
    prior_variance - prior variance characterising the prior Gaussian distribution, constant for all arms
    variance - variance of Gaussian distributions of each arm, constant for all arms, assumed known before the game begins
    n_arms - number of arms of the MAB
    time_horizon - number of time steps of the game
    arms - a list or array of arms together with their reward distribution
    
    Output : allocation sequence of arms over the time horizon
    """
    
    #### Initialisation
    
    # n[i] denotes the number of pulls of the arm i in t-1 peridos
    n = np.zeros(n_arms)
    
    # m_bar[i] denotes the mean pay-off as observed in t-1 periods
    m_bar = np.zeros(n_arms)
    
    # normalise variance
    delta_sq = variance / prior_variance
    
    # K is a tuner variable
    K = math.sqrt(2*math.pi*math.e)
    
    T_0_end = 0
    
    # rewards array
    rewards = np.zeros(time_horizon)
    
    # allocation sequence (output)
    alloc_seq = np.zeros(time_horizon)
    
    
    #### First arm
    
    # Pick the first arm based on the highest prior mean
    alloc_seq[0] = np.argmax(prior_means)
    
    # claim the reward from that arm
    # initialise a list of rewards
    all_rewards = [arms[i].draw() for i in range(n_arms)]
    rewards[0] = all_rewards[np.argmax(prior_means)]
    
    #### Iteration
    #  at each time pick an arm using Boltzmann probability distribution
    for t in range(1,time_horizon):
        
        # initialise the values of the heuristic function for each arm
        Q = np.zeros(n_arms)
        
        # initialise a list of rewards
        all_rewards = [arms[i].draw() for i in range(n_arms)]
        
        #print ("rewards:",all_rewards)
        
        for i in range(n_arms):
            
            # Compute a heuristic value for the current arm
            Q[i] = (delta_sq*prior_means[i]+n[i]*m_bar[i])/(delta_sq+n[i]) + math.sqrt(variance/(delta_sq+n[i]))*norm.ppf(1-1/(K*(t+1)))
        
        # find the minimum gap of the heuristic function value between any two arms
        delta_Q_min = find_min_diff(Q,n_arms)
        
        # define the cooling schedule
        v = delta_Q_min/(2*math.log(t+1))
        
        # select an arm with the highest probability
        P = np.exp(Q/v)/np.sum(np.exp(Q/v))
        
        """print("Q:",Q)
        print("v:",v)
        print ("P:",P)"""
        
        selected_arm = np.argmax(P)
        alloc_seq[t] = selected_arm
        
        # collect reward m_real
        rewards[t] = all_rewards[selected_arm]
        
        # update variables
        m_bar[selected_arm] = (n[selected_arm]*m_bar[selected_arm] + all_rewards[selected_arm]) / (n[selected_arm] + 1)
        n[selected_arm] += 1
    
    return alloc_seq
    


def deterministic_ucl(prior_means,prior_variance,variance,n_arms,time_horizon,arms,  n, m_bar, random_rewards=True):
    """
    Inputs: 
    prior_means - an array of prior means about the pay-offs of the arms
    prior_variance - prior variance characterising the prior Gaussian distribution, constant for all arms
    variance - variance of Gaussian distributions of each arm, constant for all arms, assumed known before the game begins
    n_arms - number of arms of the MAB
    time_horizon - number of time steps of the game
    arms - a list or array of arms together with their reward distribution
    
    Output : allocation sequence of arms over the time horizon
    """
    
    #### Initialisation
    
    # n[i] denotes the number of pulls of the arm i in t-1 peridos
    # n = np.zeros(n_arms)
    
    # m_bar[i] denotes the mean pay-off as observed in t-1 periods
    # m_bar = np.zeros(n_arms)
    
    # normalise variance
    delta_sq = variance / prior_variance
    
    # K is a tuner variable
    K = math.sqrt(2*math.pi*math.e)
    
    # rewards array
    rewards = np.zeros(time_horizon)
    
    # allocation sequence (output)
    alloc_seq = np.zeros(time_horizon)
    
    # initialise the values of the heuristic function for each arm
    Q = np.zeros(n_arms)
    
    #### Iteration
    # at each time pick the arm with maximum upper credible limit (UCL)
    for t in range(time_horizon):
        
        if random_rewards==True:
            # initialise a list of rewards
            all_rewards = [arms[i].draw() for i in range(n_arms)]
        else:
            all_rewards = arms
        
        for i in range(n_arms):
            # Compute a heuristic value for the current arm
            Q[i] = (delta_sq*prior_means[i]+n[i]*m_bar[i])/(delta_sq+n[i]) + (math.sqrt(variance/(delta_sq+n[i])))*norm.ppf(1-1/(K*(t+1)))
        
        #print("Q:",Q)
        #print('argmaxQ:',np.argmax(Q),'maxQ:',np.max(Q))
        
        # select an arm with the highest heuristic function value
        selected_arm = np.argmax(Q)
        alloc_seq[t] = selected_arm
        
        
        # collect reward m_real
        rewards[t] = all_rewards[selected_arm]
        
        #print ('all rewards:', all_rewards)
        
        # update variables
        m_bar[selected_arm] = (n[selected_arm]*m_bar[selected_arm] + all_rewards[selected_arm]) / (n[selected_arm] + 1)
        n[selected_arm] += 1
        
        #print ('m_bar:',m_bar)
        #print ('n:',n)
    
    return alloc_seq, rewards, n, m_bar
    
    

def block_ucl(prior_means,prior_variance,variance,n_arms,time_horizon, arms, random_rewards = True):
    """
    Inputs: 
    prior_means - an array of prior means about the pay-offs of the arms
    prior_variance - prior variance characterising the prior Gaussian distribution, constant for all arms
    variance - variance of Gaussian distributions of each arm, constant for all arms, assumed known before the game begins
    n_arms - number of arms of the MAB
    time_horizon - number of time steps of the game
    arms - set of arms, either input as a probability distribtion (Gaussian) or as an array of pay-offs for each arm
    random_rewards - True if the rewards are drawn i.i.d. from Gaussian distribution at each time step
    
    Output : allocation sequence of arms over the time horizon
    """
    
    
    #### Initialisation
    
    # n[i] denotes the number of pulls of the arm i in t-1 peridos
    n = np.zeros(n_arms)
    
    # m_bar[i] denotes the mean pay-off as observed in t-1 periods
    m_bar = np.zeros(n_arms)
    
    # normalise variance
    delta_sq = variance / prior_variance
    
    # K is a tuner variable
    K = math.sqrt(2*math.pi*math.e)
    
    # rewards array
    rewards = np.zeros(time_horizon)
    
    # allocation sequence (output)
    alloc_seq = np.zeros(time_horizon)
    
    # initialise the values of the heuristic function for each arm
    Q = np.zeros(n_arms)
    
    
    #### Iteration
    # at each allocation round pick the arm with maximum upper credible limit
    
    # let l be the smallest index s.t. T<2^l
    l = math.ceil(math.exp(math.log(T)/2))
    
    
    for k in range(1,l+1):
        
        # let bk be the total number of blocks in frame fk:
        bk = math.ceil((2**(k-1))/k)
        
        for r in range(1,bk+1):
            tau = 2**(k-1) + (r-1)*k
            
            for i in range(n_arms):
                # Compute a heuristic value for the current arm
                Q[i] = (delta_sq*prior_means[i]+n[i]*m_bar[i])/(delta_sq+n[i]) + math.sqrt(variance/(delta_sq+n[i]))*norm.ppf(1-1/(K*(tau)))
            
            # select an arm with the maximum value of the heuristic function
            i_hat = np.argmax(Q)
            
            #print (k,tau,2**(k)-tau)
            
            if 2**(k)-tau >= k:
                
                #print ("1")
                
                # select the same arm for the whole duration of the block
                for t in range(tau,tau+k):
                    
                    # terminate the algorithm and return the allocation sequence if the we are past the last time step
                    if t-1>=time_horizon:
                        return alloc_seq
                    
                    #print("if",t-1)
                    
                    if random_rewards == True:
                        # initialise a list of rewards
                        all_rewards = [arms[i].draw() for i in range(n_arms)]
                    else:
                        all_rewards = arms
                    
                    alloc_seq[t-1] = i_hat
                
                    #collect reward
                    rewards[t-1] = all_rewards[i_hat]
                    
                    # update variables
                    m_bar[i_hat] = (n[i_hat]*m_bar[i_hat]+all_rewards[i_hat])/(n[i_hat]+1)
                    n[i_hat] +=1
                    
            else:
                
                #print ("2")
                
                
                for t in range(tau,2**(k)):
                    
                    # terminate the algorithm and return the allocation sequence if the we are past the last time step
                    if t-1>=time_horizon:
                        return alloc_seq
                    
                    #print("else",t-1)
                    
                    if random_rewards == True:
                        # initialise a list of rewards
                        all_rewards = [arms[i].draw() for i in range(n_arms)]
                    else:
                        all_rewards = arms
                    
                    
                    alloc_seq[t-1] = i_hat
                
                    #colect reward
                    rewards[t-1] = all_rewards[i_hat]
                    
                    # update variables
                    m_bar[i_hat] = (n[i_hat]*m_bar[i_hat]+all_rewards[i_hat])/(n[i_hat]+1)
                    n[i_hat] +=1
                    
    return alloc_seq 
    
    
def generate_Bernoulli_arms(n_arms):
    return [bernoulli_arm(random.random()) for i in range(n_arms)]
    

def thompson(arms,n_arms,time_horizon,random_rewards=True):
    """
    Inputs:
    Arms 1 through n_arms, max steps time_horzin, arms object
    """
    
    #### Initialisation
    # S[i] denotes the number of successes of arm i so far
    S = np.zeros(n_arms)
    
    # F[i] denotes the number of failures of arm i so far
    F = np.zeros(n_arms)
    
    # T[i] denotes the total number of plays of arm i so far
    T = np.zeros(n_arms)
    
    # rewards array
    rewards = np.zeros(time_horizon)
    
    # allocation sequence (output)
    alloc_seq = np.zeros(time_horizon)
    
    
    #### Iteration
    for t in range(time_horizon):
        # Draw each \mu_i according to the posterior distribution
        mu = np.random.beta(1+S, 1+F)
        # play an arm according to the probability of its mean being the largest
        play = np.argmax(mu)
        alloc_seq[t] = play
        
        # Increment the total counter for the played arms
        T[play] += 1
        
        if random_rewards:
            # Observe reward
            r = arms.field[play].draw_sample()
            rewards[t] = r
        else:
            # get reward from arms
            rewards[t] = arms[play]

        # Update success counter appropriately
        S[play] += 1
        
        # Update failure counter appropriately
        F[play] += 1
        
    return alloc_seq
    

    
def discounted_thompson(arms,gamma,alpha_0,beta_0,K, T, S,F, first_iter = False):
    """
    https://arxiv.org/pdf/1707.09727.pdf
    
    Altered Thompson Sampling for restless bandits.
    
    Inputs:
        arms - arms object with parameter K
        gamma - float in (0,1], discount factor
        alpha_0, beta_0 - non-negative float, parameters for prior beta distribution
        K - int geq 2, number of arms
        T - non-negative integer, time horizon
        
    Outputs:
        allocation sequence of arms, observed rewards,
    """
    
    ## Initialisation
    if first_iter:
        # success array / failure array
        S = F = np.zeros(K)
    
    # allocation and rewards sequence
    alloc_seq = reward_seq = np.empty(0)
    
    ## Iteration
    for t in range(T):
        theta = np.random.beta(alpha_0+S, beta_0+F)
        play_arm = np.argmax(theta)
        alloc_seq = np.append(alloc_seq,play_arm)
        
        # Perform a Bernoulli trial with success probability r˜t and observe output rt
        reward = arms.field[play_arm].draw_sample()
        reward_seq = np.append(reward_seq,reward)
        
        #update variables
        S = gamma*S
        F = gamma*F
        
        S[play_arm] += reward
        F[play_arm] += (1-reward)
    
    #mean_rewards = (S)/(S+F)
    #total_plays = S+F
        
    return alloc_seq,reward_seq,S,F


    
def discounted_thompson_general(arms,gamma,alpha_0,beta_0,K, T, S,F, first_iter = False):
    """
    https://arxiv.org/pdf/1707.09727.pdf + https://arxiv.org/pdf/1111.1797.pdf
    
    Altered Thompson Sampling for restless bandits, generalised to any reward 
    distribution with support [0,1].
    
    Inputs:
        arms - arms object with parameter K
        gamma - float in (0,1], discount factor
        alpha_0, beta_0 - non-negative float, parameters for prior beta distribution
        K - int geq 2, number of arms
        T - non-negative integer, time horizon
        
    Outputs:
        allocation sequence of arms, observed rewards,
    """
    
    ## Initialisation
    if first_iter:
        # success array / failure array
        S = F = np.zeros(K)
    
    # allocation and rewards sequence
    alloc_seq = reward_seq = np.empty(0)
    
    ## Iteration
    for t in range(T):
        theta = np.random.beta(alpha_0+S, beta_0+F)
        play_arm = np.argmax(theta)
        alloc_seq = np.append(alloc_seq,play_arm)
        
        # observe reward r˜t
        reward = arms.field[play_arm].draw_sample()
        reward_seq = np.append(reward_seq,reward)
        
        # Perform a Bernoulli trial with success probability r˜t and observe output rt
        rt =  np.random.binomial(1, reward)
        
        #update variables (discounting)
        S = gamma*S
        F = gamma*F
        
        S[play_arm] += rt
        F[play_arm] += (1-rt)
    
    #mean_rewards = (S)/(S+F)
    #total_plays = S+F
        
    return alloc_seq,reward_seq,S,F
        
    
def transform_reward_seq(alloc_seq,reward_seq,n_arms):
    """Transform an allocation and reward sequence to arrays with info for each arm"""
    arms = np.zeros(n_arms)
    arms_rewards = np.zeros(n_arms)
    for k, reward in zip(alloc_seq,reward_seq):
        arms_rewards[k] = (arms[k]*arms_rewards[k]+reward)/(arms[k]+1)
        arms[k] += 1
    return arms,arms_rewards
    
