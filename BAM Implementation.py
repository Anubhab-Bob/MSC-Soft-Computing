#!/usr/bin/env python
# coding: utf-8
"""
@author: Anubhab
"""
# Driver code to implement Kosko's Bidirectional Associative Memory using custom library

# In[1]:


# import custom library
from BAM import bam


# In[2]:


# input number of pattern pairs
n = int(input("Enter the number of pattern pairs:- "))
b = bam()
# taking input pairs
for i in range(n):
    l1 = input("1st pattern of pair " + str(i+1) + " :- ")
    l2 = input("2nd pattern of pair " + str(i+1) + " :- ")
    b.create_matrix(l1, l2)	# constructs correlation matrix from input pairs

# In[4]:


# test the pattern recogniser
while True:
    choice = int(input("Enter choice of retrieval\n1: Beta\t2: Alpha\t0: Exit --> "))
    associated_pair = ""
    if choice == 0:
        print("Thank you!\n")
        break
    if choice == 1:
        alpha = input("Enter 1st pattern to retrieve its pair:- ")
        associated_pair = str(b.retrieve_beta(alpha))
    elif choice == 2:
        beta = input("Enter 2nd pattern to retrieve its pair:- ")
        associated_pair = str(b.retrieve_alpha(beta))
    # associated pair of the provided pattern
    print("The associated pair is :- \n " +','.join((filter(lambda i: i not in ['[',']',' '], associated_pair))) + "\n")
