#!/usr/bin/env python
# coding: utf-8

# # Write a python script to implement Fuzzy c-means and plot the clusters
#

# In[1]:

# import necessary libraries
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import time


# In[11]:

colors = ['y', 'ForestGreen', 'k', 'r', 'c', 'm', 'b',
          'g', 'Brown', 'orange']  # list of plotting colours
number_of_clusters = 5  # initial number of clusters

print("Generating random data..")
# Define the initial cluster centers
centers = [np.random.randint(1, 10, (2)) for _ in range(number_of_clusters)]
sigmas = [np.random.uniform(0.0, 1.0, (2)) for _ in range(number_of_clusters)]

# In[12]:

# Generate test data
np.random.seed(int(time.time()))  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
number_of_points = 200

for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(
        number_of_points) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(
        number_of_points) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(number_of_points) * i))

# Visualize the test data
fig0, ax0 = plt.subplots()
for label in range(number_of_clusters):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.',
             color=colors[label])
ax0.set_title('Test data: %d points x %d maximum clusters.' %
              (number_of_points, number_of_clusters))
print("Plotting test data (close image to continue)..")
plt.show()

# In[13]:

# Clustering our data several times, with between 2 and 10 clusters.

print("Calculating optimal number of clusters..")
# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((xpts, ypts))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    print("Clustering with %d centers.." % ncenters, end='')
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
    print("completed   FPC: %g" % fpc)
    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],
                ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()
print("Plotting generated cluster layouts (close image to continue)..")
plt.show()

# Display the best clustering generated
best_cluster = np.argmax(fpcs, axis=0) + 2
print("Best result: %d centers, FPC %g" %
      (best_cluster, fpcs[best_cluster - 2]))

# In[14]:

"""
The fuzzy partition coefficient (FPC)
-------------------------------------

The FPC is defined on the range from 0 to 1, with 1 being best. It is a metric
which tells us how cleanly our data is described by a certain model.

"""
# Display the clustering quality for different number of clusters
fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs)
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")
ax2.set_title(label="Plotting number of centers v/s FPC")
print("Plotting number of centers vs FPC (close image to continue)..")
plt.show()

# In[15]:

"""
Building the model
------------------

We know our best model has 'best_cluster' cluster centers.

"""

print("Regenerating the model with %d centers.." % best_cluster)
# Regenerate fuzzy model with 'best_cluster' cluster centers
# center ordering is random in this clustering algorithm,
# so the centers may change places
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
    alldata, best_cluster, 2, error=0.005, maxiter=1000)

# Display the n-cluster model
fig2, ax2 = plt.subplots()
ax2.set_title('Trained model')
for j in range(best_cluster):
    ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],
             alldata[1, u_orig.argmax(axis=0) == j], 'o',
             label='series ' + str(j))
ax2.legend()

print("Plotting on the trained model with the original data (close image to continue)..")
plt.show()

# In[16]:

"""
Prediction
----------

"""

print("Generating new random data..")
# Generate uniformly sampled data spread across the range [0, 10] in x and y
newdata = np.random.uniform(0, 1, (1000, 2)) * 10

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the best-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2, error=0.005, maxiter=1000)

# Display the membership values of each of 1000 points for each of the clusters
print("Membership values of 1000 test data points:")
print(u)

# Plot the classified uniform data. Note for visualization the maximum
# membership value has been taken at each point (i.e. these are hardened,
# not fuzzy results visualized) but the full fuzzy result is the output
# from cmeans_predict.
cluster_membership = np.argmax(u, axis=0)  # Hardening for visualization
fig3, ax3 = plt.subplots()
ax3.set_title('Random points classifed according to known centers')
for j in range(best_cluster):
    ax3.plot(newdata[cluster_membership == j, 0],
             newdata[cluster_membership == j, 1], 'o',
             label='series ' + str(j))
ax3.legend()
print("Plotting on the trained model with new random data (close image to continue)..")
plt.show()

print("Fuzzy C-Means Clustering Completed!")
