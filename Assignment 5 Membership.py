# -*- coding: utf-8 -*-
"""
@author: Anubhab
"""
# import necessary libraries
import matplotlib.pyplot as plt
import skfuzzy.membership as fuzzyM
import numpy as np
import random

#Triangular MF
def triangular_MF(points,m_min=0, m_max=10):
    p= np.arange(m_min, m_max+1)  # range of values
    membership_func = fuzzyM.trimf(p, points)  # triangular MF
    # Display the membership function
    plt.plot(p, membership_func, 'b', label='Triangular MF')
    plt.legend()
    plt.title(label='Triangular MF Plot')
    plt.show()

#Trapezoidal MF
def trapezoidal_MF(points,m_min=0, m_max=10):
    p= np.arange(m_min, m_max+1)  # range of values
    membership_func = fuzzyM.trapmf(p, points)  # trapezoidal MF
    # Display the membership function
    plt.plot(p, membership_func, 'black', label='Trapezoidal MF')
    plt.legend()
    plt.title(label='Trapezoidal MF Plot')
    plt.show()

#Gaussian MF
def gaussian_MF(points,m_min=0, m_max=10):
    p= np.arange(m_min, m_max+1)  # range of values
    membership_func = fuzzyM.gaussmf(p, np.mean(points),np.std(points))  # gaussian MF
    # Display the membership function
    plt.plot(p, membership_func, 'brown', label='Gaussian MF')
    plt.legend()
    plt.title(label='Gaussian MF Plot')
    plt.show()

#Gbell MF
def gBell_MF(dim, m_min=0, m_max=10):
    p= np.arange(m_min, m_max+1)  # range of values
    membership_func = fuzzyM.gbellmf(p, dim[0],dim[1],dim[2])  # Gbell MF
    # Display the membership function
    plt.plot(p, membership_func, 'magenta', label='GBell MF')
    plt.legend()
    plt.title(label='Generalised Bell MF Plot')
    plt.show()

#Sigmoid MF
def sigmoid_MF(center, m_min=0, m_max=10):
    p= np.arange(m_min, m_max+1)  # range of values
    membership_func = fuzzyM.sigmf(p, center, 2)  # Sigmoid MF
    # Display the membership function
    plt.plot(p, membership_func, 'violet', label='Sigmoid MF')
    plt.legend()
    plt.title(label='Sigmoid MF Plot')
    plt.show()

# Driver code
while True:
    print("\n1. Triangular MF\n2. Trapezoidal MF")
    print("3. Gaussian MF\n4. GBell MF\n5. Sigmoid MF")
    print("0. Exit")
    choice = int(input("Enter choice :"))
    if choice == 1:
        triangular_MF(sorted(random.sample(range(1, 10), 3)))
    elif choice == 2:
        trapezoidal_MF(sorted(random.sample(range(1, 10), 4)))
    elif choice == 3:
        gaussian_MF([random.uniform(-5,5) for _ in range(10000)],m_min=-10,m_max=10)
    elif choice == 4:
        gBell_MF((1.5, 5, 5))
    elif choice == 5:
        sigmoid_MF(random.randint(2, 8))
    else:
        print("Thank You!\n")
        break