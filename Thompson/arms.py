# -*- coding: utf-8 -*-
"""
Arms of multi-armed bandits
"""

import numpy as np
import scipy
from random import gauss
from numpy.random import standard_normal
from scipy.special import erf
import random

np.random.seed(None)
random.seed()

oo = float('+inf') #+infinity

#: Default value for the variance of a [0, 1] Gaussian arm
VARIANCE = 0.05


def phi(xi):
    """
    the probability density function of the standard normal distribution
    """
    return np.exp(- 0.5 * xi**2) / np.sqrt(2. * np.pi)


def Phi(x):
    """
    It is the probability density function of the standard normal distribution
    """
    return (1. + erf(x / np.sqrt(2.))) / 2.


class Arm(object):
    """ Base class for an arm class."""
    
    def __init__(self, lower=0., amplitude=1.):
        """ Base class for an arm class."""
        self.lower = lower  #: Lower value of rewards
        self.amplitude = amplitude  #: Amplitude of value of rewards
        self.min = lower  #: Lower value of rewards
        self.max = lower + amplitude  #: Higher value of rewards

    # make this an attribute
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        if hasattr(self, 'lower') and hasattr(self, 'amplitude'):
            return self.lower, self.amplitude
        elif hasattr(self, 'min') and hasattr(self, 'max'):
            return self.min, self.max - self.min
        else:
            raise NotImplementedError("This method lower_amplitude() has to be implemented in the class inheriting from Arm.")

                                      
    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dir__)

                                                                            
    #### Random samples
    def draw(self, t=None):
        """ Draw one random sample."""
        raise NotImplementedError("This method draw(t) has to be implemented in the class inheriting from Arm.")

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        raise NotImplementedError("This method draw_nparray(t) has to be implemented in the class inheriting from Arm.")

    #### Lower bound
    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        raise NotImplementedError("This method kl(x, y) has to be implemented in the class inheriting from Arm.")

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - mu) / KL(mu, mumax). """
        raise NotImplementedError("This method oneLR(mumax, mu) has to be implemented in the class inheriting from Arm.")

    @staticmethod
    def oneHOI(mumax, mu):
        """ One term for the HOI factor for this arm."""
        return 1 - (mumax - mu)

        
        
class Gaussian(Arm):
    """
    Gaussian distributed arm, possibly truncated.
    Default is to truncate into [0, 1] (so Gaussian.draw() is in [0, 1]).
    """

    def __init__(self, mu, sigma=VARIANCE, mini=0, maxi=1):
        """New arm."""
        self.mu = self.mean = mu  #: Mean of Gaussian arm
        assert sigma > 0, "Error, the parameter 'sigma' for Gaussian arm has to be > 0."
        self.sigma = sigma  #: Variance of Gaussian arm
        assert mini <= maxi, "Error, the parameter 'mini' for Gaussian arm has to < 'maxi'."  # DEBUG
        self.min = mini  #: Lower value of rewards
        self.max = maxi  #: Higher value of rewards
        real_mean = mu + sigma * (phi(mini) - phi(maxi)) / (Phi(maxi) - Phi(mini))

        
    #### Random samples
    def draw_sample(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return min(max(gauss(self.mu, self.sigma), self.min), self.max)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return np.minimum(np.maximum(self.mu + self.sigma * standard_normal(shape), self.min), self.max)

    
    #### Printing
    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return self.min, self.max - self.min

    def __str__(self):
        return "Gaussian"

    def __repr__(self):
        return "N({:.3g}, {:.3g})".format(self.mu, self.sigma)

    #### Lower bound - NEED TO IMPLEMENT KULLBACK-LEIBLER DIVERGENCE FOR GAUSSIAN DISTRIBUTION FIRST
    if False:
        def kl(self, x, y):
            """ The kl(x, y) to use for this arm."""
            return klGauss(x, y, self.sigma)

        def oneLR(self, mumax, mu):
            """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - mu) / KL(mu, mumax). """
            return (mumax - mu) / klGauss(mu, mumax, self.sigma)
    
    def oneHOI(self, mumax, mu):
        """ One term for the HOI factor for this arm."""
        return 1 - (mumax - mu) / self.max


class Gaussian_0_1(Gaussian):
    """ Gaussian distributed arm, truncated to [0, 1]."""
    def __init__(self, mu, sigma=0.05, mini=0, maxi=1):
        super(Gaussian_0_1, self).__init__(mu, sigma=sigma, mini=mini, maxi=maxi)


#: Default value for the variance of an unbounded Gaussian arm
UNBOUNDED_VARIANCE = 1


class UnboundedGaussian(Gaussian):
    """ Gaussian distributed arm, not truncated, ie. supported in (-oo,  oo)."""

    def __init__(self, mu, sigma=UNBOUNDED_VARIANCE):
        """New arm."""
        super(UnboundedGaussian, self).__init__(mu, sigma=sigma, mini=-oo, maxi=oo)

    def __str__(self):
        return "UnboundedGaussian"

    #### Random samples

    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return gauss(self.mu, self.sigma)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return self.mu + self.sigma * standard_normal(shape)

    def __repr__(self):
        return "N({:.3g}, {:.3g})".format(self.mu, self.sigma)


class truncated_gaussian_arms(object):
    def __init__(self,n):
        self.n = n
        self.field = [Gaussian_0_1(random.random()) for i in range(self.n)]
        self.field_exp = [arm.mu for arm in self.field]

    def add_noise(self):
        self.field = [bernoulli_arm(bound_r(arm.expectation+np.random.normal(0,0.05))) for arm in self.field]

        
#A = truncated_gaussian_arms(10)
#print (A.field[0].draw())

class bernoulli_arm(object):
    def __init__(self,p):
        self.p=p
        self.expectation = p
        
    '''
    pull an arm according to Bernoulli, return 1 or 0
    '''
    def draw_sample(self):
        # random.random() Return the next random floating point number in the range [0.0, 1.0)
        return float(random.random()<self.p)
    
        
class bernoulli_arms(object):
    def __init__(self,n):
        self.n = n
        self.field = [bernoulli_arm(random.random()) for i in range(self.n)]
        self.field_exp = [arm.expectation for arm in self.field]

    def add_noise(self):
        self.field = [bernoulli_arm(bound_r(arm.expectation+np.random.normal(0,0.05))) for arm in self.field]

        
def bound_r(r):
    if r>1:
        return 1
    elif r<0:
        return 0
    else:
        return r