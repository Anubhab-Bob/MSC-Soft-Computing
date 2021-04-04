# -*- coding: utf-8 -*-
"""
@author: Anubhab
"""
# import necessary libraries
import random
from termcolor import colored

#initialize population
best=-100000
populations =([[random.randint(0,1) for x in range(6)] for i in range(4)])
parents=[]
new_populations = []
print("Initial Population: ", populations)

# function to calculate fitness score
def f_score(clist):
    final = []
    for i in range(len(clist)) :
        chromosome_value=0
        for j in range(5,0,-1) :
            chromosome_value += clist[i][j]*(2**(5-j))
            chromosome_value = -1*chromosome_value if clist[i][0]==1 else chromosome_value
            final.append(-(chromosome_value)**2 + 3 )
    return final

# function for selecting parents
def selectparent():
    global parents,populations
    scores =f_score(populations)
    temp=[]
    for i in range(len(scores)):
        temp.append(abs(scores[i]))
    maxi=max(temp)
    for i in range(len(scores)):
        scores[i]=scores[i]+maxi+1
    tot=sum(scores)
    probabilty=[]
    for i in range(4):
        probabilty.append(abs(scores[i]/tot))
    probability, populations, scores = zip(*sorted(zip(probabilty, populations, scores), reverse=True ))
    probabilty=list(probability)
    populations=list(populations)
    print("\nPopulations \t\tFitness Scores \t\t Probabilities")
    for i in range(len(populations)):
        print("{0} \t\t {1} \t\t {2} ".format(populations[i],scores[i],probabilty[i]))
    parents=populations[0:2]
    print("\nParents: ",colored(parents[0],'red'),"\t",colored(parents[1],'blue'))

# invoke selectparent() to select the parents
selectparent()

# function for single-point crossover
def crossover() :
    global parents,populations,best
    children=[]
    print("\nChildren:")
    for i in range(0,5):
        cross_point=i
        children.append(parents[0][0:cross_point +1] +parents[1][cross_point+1:6])
        print("\n[",colored(parents[0][0:cross_point +1],'red'),colored(parents[1][cross_point+1:6],'blue'),"]",end="\t")
        children.append(parents[1][0:cross_point +1] +parents[0][cross_point+1:6])
        print("[",colored(parents[1][0:cross_point +1],'blue'),colored(parents[0][cross_point+1:6],'red'),"]")
    scores=f_score(children)
    scores, children = zip(*sorted(zip(scores, children), reverse=True ))
    children=list(children)
    scores=list(scores)
    if scores[0]>best:
        best=scores[0]
    populations.clear()
    populations=children[:]

# initiate crossover
crossover()

# function to introduce mutation
def mutation(k: int) :
    global populations
    mute = random.randint(0,20)
    if mute == 15 :
        x=random.randint(0,len(populations)-1)
        y = random.randint(1,5)
        print("Mutation Occured!")
        populations[x][y] = 1-populations[x][y]
    print("\n\nPopulation of generation: ",k)

# perform mutation
mutation(0)

#Main Driver function
for i in range(20) :
    selectparent()
    crossover()
    mutation(i+1)
print("\n\n\nBest value of x :",best)
