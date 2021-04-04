# -*- coding: utf-8 -*-
"""
@author: Anubhab
"""
# import necessary libraries
import skfuzzy as sk
import numpy as np

# 2 random fuzzy relations
fuzzy1 = np.round(np.array(np.random.rand(2,2)), decimals=3)  # fuzzy relation with 2x2 dimension
fuzzy2 = np.round(np.array(np.random.rand(2,3)), decimals=3)  # fuzzy relation with 2x3 dimension

# Calculate Max-Product Composition relation
mx_pd_fuzzyComp = np.round(sk.fuzzymath.maxprod_composition(fuzzy1, fuzzy2),decimals=3)

print("Fuzzy Relation Matrix 1 :\n",str(fuzzy1))
print("Fuzzy Relation Matrix 2 :\n",fuzzy2)
print ("Fuzzy Max-Product composition of the input fuzzy relations -->\nR1oR2 => Max-Product :\n",mx_pd_fuzzyComp)