#!/usr/bin/env python
# coding: utf-8
"""
@author: Anubhab
"""
# A custom library to implement Kosko's Bidirectional Associative Memory.

# In[1]:
import numpy as np
# bam class
class bam(object):

    def __init__(self):
        # constructor
        self.M = None

    def create_matrix(self, a, b):
       # transpose a and multiply with b
        a, b = a.replace("0", "-1"), b.replace("0", "-1")
        a, b = list(map(int, a.split(','))), list(map(int, b.split(',')))
        a = np.array(a).reshape(len(a), 1)
        self.store_matrix(np.multiply(a, b))

    def store_matrix(self, matrix):
       # summation of the matrices obtained after multiplication
        if self.M is None:
            self.M = matrix
        else:
            self.M += matrix

    def replacer(self, array):
	# replaces values in patterns
        array[array > 0], array[array < 0] = 1, -1
        return array

    def retrieve_beta(self, alpha):
    # retrieves the pattern pair
        # alpha is the bipolar input pattern
        alpha = alpha.replace("0", "-1")
        alpha = list(map(int, alpha.split(',')))
        alpha = np.array(alpha).reshape(1, len(alpha))
        beta = self.replacer(np.dot(alpha, self.M))
        beta1 = None
        while not (beta == beta1).all():
            alpha1 = self.replacer(np.dot(beta, np.transpose(self.M)))
            beta1 = self.replacer(np.dot(alpha1, self.M))
        beta[beta < 0] = 0
        return beta

    def retrieve_alpha(self, beta):
    # retrieves the pattern pair
        # beta is the bipolar input pattern
        beta = beta.replace("0", "-1")
        beta = list(map(int, beta.split(',')))
        beta = np.array(beta).reshape(1, len(beta))
        alpha = self.replacer(np.dot(beta, self.M.T))
        alpha1 = None
        while not (alpha == alpha1).all():
            beta1 = self.replacer(np.dot(alpha, self.M))
            alpha1 = self.replacer(np.dot(beta1, self.M.T))
        alpha[alpha < 0] = 0
        return alpha