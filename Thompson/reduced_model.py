import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import random
import networkx as nx
import math
from scipy import stats
import pickle
import community
import copy
import math
import matplotlib.animation as animation
from collections import Counter
from scipy.cluster.hierarchy import dendrogram, linkage, fclusterdata
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import time
from IPython.display import clear_output
import datetime
from scipy.spatial.distance import pdist, squareform
import scipy.integrate as ODE

random.seed()

def dot_product_coordinates(pos,coeffs):
    z = np.array([pos[i]*coeffs[i] for i in range(np.size(coeffs))])
    return np.sum(z,axis=0)

def transform_clusters(clusters):
    d = dict()
    for e in np.unique(clusters):
        d[e] = [i for i,j in enumerate(clusters) if j==e]
    return d

def discrete_step_number(x):
    return 1 if x>=0 else -1

def discrete_step(ar):
    return np.where(ar>=0,1,-1)

def dist_arr(pos,ar):
    dist = lambda p1,p2: math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return np.array([dist(pos,ar[i]) for i in range(np.size(ar,0))])

def sec_to_string(seconds) :
    return str(datetime.timedelta(seconds=seconds))

def rk4(f, t0, y0, t1, n):
    """Runge-Kutta 4 method"""
    vt = [0] * (n + 1)
    vy = np.zeros((y0.size,n + 1))
    h = (x1 - x0) / float(n)
    vt[0] = t = t0
    vy[,0] = y = y0
    for i in range(1, n + 1):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        vt[i] = x = x0 + i * h
        vy[,i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return vt, vy

class progress_bar:
    
    def __init__(self,n_max):
        self.n_max = n_max
    
    def update_progress(self,progress):
        bar_length = self.n_max
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
        if progress < 0:
            progress = 0
        if progress >= 1:
            progress = 1
        return progress
            
    def display(self, progress):
        progress = self.update_progress(progress)
        block = int(round(self.n_max * progress))
        clear_output(wait = True)
        text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (self.n_max - block), progress * 100)
        print(text)
        
class roulette_wheel(object):
    
    def __init__(self,pop):
        self.pop = pop
        self.pop_fitness = self.pop_fitness()
        self.probabilities = self.get_probability_list()
    
    def pop_fitness(self):
        pop_fitness = dict()
        for n in range(len(self.pop)):
            pop_fitness[n] = self.pop.payoffs[n]
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
            for i in range(len(self.pop)):
                if r <= self.probabilities[i]:
                    chosen.append(i)
                    break
        return chosen
    
class pop:
    
    def __init__(self, n_res, p_coop, q, n_mut, beta, gamma, alpha, payoff_func, power_func):
        self.n_resident = n_res
        self.p_coop = p_coop
        self.q = q
        self.n_mutants = n_mut
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.payoff_func = payoff_func
        self.power_func = power_func
        
    def __len__(self):
        return self.n_resident + self.n_mutants

    def init_players_resident(self,p_coop = 1, clustering = False, n_clusters = 4, group = False, radius = 0.5):
        self.p_sharing = np.array([self.p_coop for i in range(self.n_resident)])
        if clustering:
            p = q = []
            for n in range(self.n_resident):
                m = n%n_clusters
                if m==0:
                    x,y = 0.1+np.random.uniform(-0.05,0.05), 0.1+np.random.uniform(-0.05,0.05)
                elif m==1:
                    x,y = 0.9+np.random.uniform(-0.05,0.05), 0.1+np.random.uniform(-0.05,0.05)
                elif m==2:
                    x,y = 0.9+np.random.uniform(-0.05,0.05), 0.9+np.random.uniform(-0.05,0.05)
                else:
                    x,y = 0.1+np.random.uniform(-0.05,0.05), 0.9+np.random.uniform(-0.05,0.05)
                p.append(x)
                q.append(y)
            self.pos = np.array(list(zip(p,q)))
        elif group:
            radi = np.random.uniform(0,radius,self.n_resident)
            thetas = np.random.uniform(-math.pi,math.pi,self.n_resident)
            x, y = radi*np.cos(thetas), radi*np.sin(thetas)
            self.pos = np.array(list(zip(x,y)))
        else:
            x,y = np.random.uniform(0,1,self.n_resident), np.random.uniform(0,1,self.n_resident)
            self.pos = np.array(list(zip(x,y)))
            
    def add_mutants(self, group = False, radius = 0.5, outer_ring = False, locs=None):
        q_sharing = np.array([self.q for i in range(self.n_mutants)])
        self.p_sharing = np.append(self.p_sharing, q_sharing)
        if locs==None:
            if group:
                if outer_ring:
                    thetas = np.random.uniform(-math.pi,math.pi,self.n_mutants)
                    x, y = radius*np.cos(thetas), radius*np.sin(thetas)
                else:
                    radi = np.random.uniform(0,radius,self.n_mutants)
                    thetas = np.random.uniform(-math.pi,math.pi,self.n_mutants)
                    x, y = radi*np.cos(thetas), radi*np.sin(thetas)
            else:
                x,y = np.random.uniform(0,1,self.n_mutants), np.random.uniform(0,1,self.n_mutants)
            self.pos = np.concatenate((self.pos,np.array(list(zip(x,y)))))
        else:
            for loc in locs:
                self.pos = np.concatenate(self.pos,loc)
            
    def create_game_matrix(self):
        """Create a matrix of strategies"""
        beta = self.beta
        dm = np.exp(squareform(pdist(self.pos))*(math.log(beta)/math.sqrt(2)))
        self.dm = dm
        gm = (dm.T * self.p_sharing).T
        np.fill_diagonal(gm, 0)
        self.current_coop = gm
        self.coop_matrix = gm
        
    def movement(self, v_max = 1, i):
        old_pos = self.pos[i]
        moves = self.moves
        neighbors = np.asarray([int(j) for j in range(self.n_resident + self.n_mutants) if j!=i])
        weights_neighbours = self.dm[i,neighbors]
        div = self.p_sharing[neighbors] - self.p_sharing[i]
        #div = self.coop_matrix[neighbors,i] - self.coop_matrix[i,neighbors]
        sgn = discrete_step(div)
        #print (sgn)
        func_form = (1+abs(div))**self.alpha
        #print (old_pos)
        decay_dist = 1/(1+dist_arr(old_pos[i],old_pos[neighbors]))**self.gamma
        weights = sgn*func_form
        w_ij = weights*decay_dist
        #print ('w_ij',w_ij)
        a_ij = w_ij/np.sum(np.absolute(w_ij))
        a_ij = a_ij*v_max
        #print ('v_max',v_max)
        #print ('a_ij',a_ij)
        pm = moves[i,neighbors]
        #print ('pm',pm)
        #print (dot_product_coordinates(pm,a_ij))
        d_p = dot_product_coordinates(pm,a_ij)
        return d_p
    
    def f(t,y):
        
        
    def update_pos(self, v_max = 1, eta = 0.01, normalise = True, limit_interactions = False, interaction_radius = 0.5):
        old_pos = self.pos
        self.pos = []
        self.moves = moves = np.asarray([[(p2-p1)/np.linalg.norm(p2-p1) for p2 in old_pos] for p1 in old_pos])
        for i in range(self.n_resident + self.n_mutants):
            neighbors = np.asarray([int(j) for j in range(self.n_resident + self.n_mutants) if j!=i])
            weights_neighbours = self.dm[i,neighbors]
            div = self.p_sharing[neighbors] - self.p_sharing[i]
            #div = self.coop_matrix[neighbors,i] - self.coop_matrix[i,neighbors]
            sgn = discrete_step(div)
            #print (sgn)
            func_form = (1+abs(div))**self.alpha
            #print (old_pos)
            decay_dist = 1/(1+dist_arr(old_pos[i],old_pos[neighbors]))**self.gamma
            weights = sgn*func_form
            w_ij = weights*decay_dist
            #print ('w_ij',w_ij)
            a_ij = w_ij/np.sum(np.absolute(w_ij))
            a_ij = a_ij*v_max
            #print ('v_max',v_max)
            #print ('a_ij',a_ij)
            pm = moves[i,neighbors]
            #print ('pm',pm)
            #print (dot_product_coordinates(pm,a_ij))
            d_p = dot_product_coordinates(pm,a_ij) + np.random.uniform(-eta,eta,2)
            # Perform Runga-Kutter numerical integration of the movement ODE
            RK45(simplem,t0 = 0.0, y0 = old_pos[i], t_bound = 1.0)
            #print ('d_p',d_p)
            #print (np.linalg.norm(d_p))
            #if normalise:
            #    d_p = d_p/(np.linalg.norm(d_p))
            self.pos.append(old_pos[i] + d_p)
        #print (self.pos)
        self.pos = np.asarray(self.pos)
            
    def pay_off(self, treshold = 0.25):
        average_sharing = [np.mean(np.array([self.coop_matrix[n][i] for n in range(len(self)) if n!=i])) for i in range(len(self))]
        if self.payoff_func == 'quorum':
            self.payoffs = np.power(average_sharing,self.power_func)/(np.power(average_sharing,self.power_func)+np.power(np.full(len(self),treshold),self.power_func))
        elif self.payoff_func == 'power':
            self.payoffs = np.power(average_sharing,self.power_func)
        elif self.pay_off_func == 'arctan':
            self.payoffs = np.power(np.arctan(average_sharing),self.power_func)
        
    def selection(self):
        RL = roulette_wheel(self)
        self.parents = RL.roulette_wheel_pop()
        
    def reproduction(self, mutation_sharing = False):
        self.p_sharing = self.p_sharing[np.asarray(self.parents)]
        if mutation_sharing:
            self.p_sharing = np.clip(self.p_sharing + np.random.normal(0,0.005,len(self)),a_min=0,a_max=1)
        pos = np.array([self.pos[i] for i in self.parents])
        self.pos = pos + np.random.normal(0,0.005,(len(self),2))
        self.create_game_matrix()
        self.pay_off()
        
    def silloutte_k_means(self):
        """Perform the silloutte k-means clustering analysis"""
        range_n_clusters = range(2,15)
        #X = self.get_locations_all_list()
        X = self.pos
        silhouettes = np.empty(0)
        clusterers=[]
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            clusterers.append(clusterer)
            cluster_labels = clusterer.fit_predict(X)
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouettes = np.append(silhouettes, silhouette_avg)
        k_best = np.argmax(silhouettes)
        clusterer = clusterers[k_best]
        kmeans = clusterer.fit(X)
        labels = kmeans.labels_
        clusters = transform_clusters(labels)
        info_sharing_clusters = []
        for key,value in clusters.items():
            if len(value)==0:
                assert ValueError ('An empty cluster found')
            elif len(value)==1:
                info_sharing_clusters.append(self.p_sharing[value[0]])
            else:
                info_sharing_clusters.append(np.mean(np.array([self.coop_matrix[i][j] for i in value for j in value if i!=j])))
        unique_elements, counts_elements = np.unique(labels, return_counts=True)
        return np.size(unique_elements), np.max(counts_elements), counts_elements, info_sharing_clusters, labels
    
    def get_locations_p(self):
        loc = self.pos[np.where(self.p_sharing>self.q)]
        #loc = [self.pos[n] for n in range(len(self)) if self.p_sharing[n]>self.q]
        return loc
    
    def get_locations_q(self):
        loc = self.pos[np.where(self.p_sharing<=self.q)]
        #loc = [self.pos[n] for n in range(len(self)) if self.p_sharing[n]<=self.q]
        return loc
    
    
class Sim_mutant_resident:
    
    def __init__(self, n_iter, n_res, p_coop, q, n_mut, beta = 0.2, gamma=2, alpha = 2, normalise = True, clustering = False, n_clusters = 4, payoff_func = 'quorum', power_func = 4, group = False, radius = 0.5, outer_ring = False):
        self.n_iter = n_iter
        self.population = self.init_pop(n_res, p_coop, q, n_mut, beta, gamma, alpha, clustering, n_clusters, payoff_func, power_func, group, radius, outer_ring)
        self.normalise = normalise
        
    def init_pop(self, n_res, p_coop, q, n_mut, beta, gamma, alpha, clustering = False, n_clusters = 4, payoff_func = 'quorum', power_func = 2, group = False, radius = 0.5, outer_ring = False):
        population = pop(n_res, p_coop, q, n_mut, beta, gamma, alpha, payoff_func, power_func)
        population.init_players_resident(p_coop, clustering, n_clusters, group, radius)
        population.add_mutants(group, radius, outer_ring)
        population.create_game_matrix()
        return population
    
    def loop_pos(self, selection = False, steps = 20, v_max = 0.05, eta = 0.001, limit_interactions = False, interaction_radius = 0.5, treshold = 0.25):
        start = int(round(time.time()))
        loc_p=dict()
        loc_q=dict()
        bar = progress_bar(self.n_iter)
        for t in range(self.n_iter):
            loc_p[t] = self.population.get_locations_p()
            loc_q[t] = self.population.get_locations_q()
            self.population.update_pos(v_max, eta,self.normalise, limit_interactions, interaction_radius)
            if selection:
                self.population.pay_off(treshold)
                if t % steps == 0:
                    self.population.selection()
                    self.population.reproduction()
            bar.display((t+1)/self.n_iter)
        print ('Duration:',sec_to_string(int(round(time.time())) - start))
        return loc_p, loc_q
    
    def loop(self, selection = False, steps = 20, mutation_sharing = False, v_max = 0.05, eta = 0.001, limit_interactions = False, interaction_radius = 0.5, treshold = 0.25):
        start = int(round(time.time()))
        loc_p=dict()
        loc_q=dict()
        n_clusters=[]
        max_cluster=[]
        size_clusters=[]
        info_sharing_clusters=[]
        strategies = []
        bar = progress_bar(self.n_iter)
        for t in range(self.n_iter):
            loc_p[t] = self.population.get_locations_p()
            loc_q[t] = self.population.get_locations_q()
            n_clusters.append(self.population.silloutte_k_means()[0])
            max_cluster.append(self.population.silloutte_k_means()[1])
            size_clusters.append(self.population.silloutte_k_means()[2])
            info_sharing_clusters.append(self.population.silloutte_k_means()[3])
            strategies.append(self.population.p_sharing)
            self.population.update_pos(v_max,eta,self.normalise, limit_interactions, interaction_radius)
            if selection:
                self.population.pay_off(treshold)
                if t % steps == 0:
                    self.population.selection()
                    self.population.reproduction(mutation_sharing)
            bar.display((t+1)/self.n_iter)
        print ('Duration:',sec_to_string(int(round(time.time())) - start))
        return loc_p, loc_q, n_clusters, max_cluster, size_clusters, info_sharing_clusters, strategies
    
    def plot_results(self, selection=False, steps = 20, mutation_sharing = False, v_max = 0.05, eta = 0.001, limit_interactions = False, interaction_radius = 0.5, treshold = 0.25):
        loc_p, loc_q, n_clusters, max_cluster, size_clusters, info_sharing_clusters, sharing_strategies = self.loop(selection, steps, mutation_sharing, v_max, eta, limit_interactions, interaction_radius, treshold)
        clusters = np.array([list(i) for i in size_clusters])
        sharing = np.array([i for i in info_sharing_clusters])
        strategies = np.array([list(i) for i in sharing_strategies])
        t = [i for i in range(1,self.n_iter+1)]
        
        fig = plt.figure(figsize=(10,12))
        
        plt.subplot(5, 2, 1)
        plt.plot(t,n_clusters)
        plt.title('Number of clusters')
        
        plt.subplot(5, 2, 2)
        plt.plot(t,max_cluster)
        plt.title('Size of the largest cluster')
        
        plt.subplot(5,2,3)
        for k in t:
            plt.scatter([k for i in range(len(clusters[k-1]))],clusters[k-1], marker='o', s=2, color='k')
        plt.title('Evolution of cluster sizes')
        
        plt.subplot(5,2,4)
        for k in t:
            plt.scatter([k for i in range(len(sharing[k-1]))],sharing[k-1], marker='o', s=2, color='k')
        plt.title('Evolution of sharing rates in within clusters')
        
        plt.subplot(5,2,5)
        for k in t:
            plt.scatter([k for i in range(len(strategies[k-1]))],strategies[k-1], marker='o', s=2, color='k')
        plt.title('Evolution of strategies')
        
        fig.tight_layout()
        plt.show()                     
        
        return loc_p, loc_q
