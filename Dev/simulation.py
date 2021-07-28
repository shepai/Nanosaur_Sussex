import matplotlib.pyplot as plt
import math
import time
import random
import copy

import cv2
import numpy as np
import copy
import torch


plt.ylim((-50,50))
plt.xlim((-50,50))

def getBump(a,points):
    if [a.x,a.y] in points:
        return True
    return False
def fitness(numBumps,timeGiven):
    #print((timeGiven-numBumps)/20)
    return (timeGiven-numBumps)/20

def mutation(gene, mean=0, std=0.1):
    gene = gene + np.random.normal(mean, std, size=gene.shape) #mutate the gene via normal 
    # constraint
    gene[gene > 4] = 4
    gene[gene < -4] = -4
    return gene

def crossover(loser, winner, p_crossover=0.5):
    for i,gene in enumerate(winner):
      if random.random() <= p_crossover:
        loser[i] = winner[i]
    return loser
class Agent:
    def __init__(self, num_input, num_hiddenLayer, num_output):
        self.num_input = num_input  #set input number
        self.num_output = num_output #set ooutput number
        self.num_genes = (num_input * num_hiddenLayer) + (num_output) + (num_hiddenLayer*num_output)
        self.num_hidden=num_hiddenLayer
        self.weights = None
        self.weights2=None
        self.bias = None

    def set_genes(self, gene):
        weight_idxs = self.num_input * self.num_hidden #size of weights to hidden
        weights2_idxs = self.num_hidden * self.num_output + weight_idxs #size and position
        bias_idxs = weight_idxs + weights2_idxs + self.num_output #sizes of biases
        w = gene[0 : weight_idxs].reshape(self.num_hidden, self.num_input)   #merge genes
        w2 = gene[weight_idxs : weights2_idxs].reshape(self.num_output, self.num_hidden)   #merge genes
        b = gene[weights2_idxs: bias_idxs].reshape(self.num_output,) #merge genes
        self.weights = torch.from_numpy(w) #assign weights
        self.weights2 = torch.from_numpy(w2) #assign weights
        self.bias = torch.from_numpy(b) #assign biases

    def forward(self, x):
        x = torch.from_numpy(x.flatten()/255).unsqueeze(0)
        x=torch.mm(x, self.weights.T) #first layer
        return torch.mm(x,self.weights2.T) + self.bias #secon layer
        
    def get_action(self, x):
        #print(self.forward(x))
        a=[]
        for i in list(self.forward(x)[0]):
            if i > 0.05: #foward
                a.append(1)
            elif i <-0.05:
                a.append(-1)
            else: #backward
                a.append(0)
        
        return a
    
class agent:
    def __init__(self):
        self.x=0
        self.y=0
        self.vector=[0,0]
    def set_vector(self,x,y):
        self.vector=[x,y]
        self.x+=x
        self.y+=y
    def get_image(self,points,r=8):
        #get array facing the vector
        #(x-self.x)^2+ (y-self.y)^2 = 25
        arr=[]
        for j in range(r):
            toAdd=0
            for i in range(r):
                if [self.x+j-i,self.y+i] in points: toAdd=1
            arr.append(toAdd)
        for j in range(r):
            toAdd=0
            for i in range(r):
                if [self.x-i,self.y+j-i] in points: toAdd=1
            arr.append(toAdd)
        for j in range(r):
            toAdd=0
            for i in range(r):
                if [self.x-j-i,self.y-i] in points: toAdd=1
            arr.append(toAdd)
                
        for j in range(r):
            toAdd=0
            for i in range(r):
                if [self.x-j+i,self.y-j+i] in points: toAdd=1
            arr.append(toAdd)
        return np.array(arr)



def show_points(points,agent):
    plt.cla()
    for p in points:
        plt.scatter(p[0],p[1],c="y")
    plt.scatter(agent.x,agent.y,c="r")
    plt.pause(0.1)
##generate enviroment
points=[]
for i in range(600):
    randPoint=[random.randint(-50,50),random.randint(-50,50)]
    points.append(copy.deepcopy(randPoint))
    plt.scatter(randPoint[0],randPoint[1],c="y",s=20)
if [0,0] in points:
    points.remove([0,0])
plt.scatter(0,0,c="r",s=20)
a=agent()
pixels=a.get_image(points).shape
ag=Agent(pixels[0],10,2) #h*w inputs for pixels
pop_size=10
gene_pop=[]

#check if genes exist in file
try:
    for i in range(pop_size):
        f=np.load(str(i)+'.npy')
        gene_pop.append(copy.deepcopy(f))
except: #otherwise create
    for i in range(pop_size): #vary from 10 to 20 depending on purpose of robot
      gene_pop.append(np.random.normal(0, 0.1, (ag.num_genes)))#create
      
Generations=500
print("Begin")
fitness_index=[0 for i in range(pop_size)]

TIME=20

for gen in range(Generations):
    a=agent()
    #perform Reinforcement learning 
    current=a.get_image(points)
    print("Gen",gen)
    #Apply microbial
    n1=random.randint(0,len(gene_pop)-1)
    n2=random.randint(0,len(gene_pop)-1)
    g1=mutation(gene_pop[n1])
    g2=mutation(gene_pop[n1])

    #apply gene 1
    print("gene",n1,"trial")
    fitness1=0
    t=time.time()
    currentT=time.time()
    ag.set_genes(g1) #set the genes
    while currentT-t<TIME and getBump(a,points)==False: #give 20 seconds for trial
        currentT=time.time()
        current=a.get_image(points)
        action=ag.get_action(current)
        a.set_vector(action[0],action[1])
        show_points(points,a)
        prvs = copy.deepcopy(current)
    fitness1=fitness(1,currentT-t)
    if fitness1> fitness_index[n1]: fitness_index[n1]=fitness1 #keep track of fitnesses
    #apply gene 2
    print("gene",n2,"trial")
    fitness2=0
    t=time.time()
    currentT=time.time()
    ag.set_genes(g2) #set the genes
    while currentT-t<TIME and getBump(a,points)==False: #give 20 seconds for trial
        currentT=time.time()
        current=a.get_image(points)
        action=ag.get_action(current)
        a.set_vector(action[0],action[1])
        show_points(points,a)
        prvs = copy.deepcopy(current)
    fitness2=fitness(1,currentT-t)
    if fitness2> fitness_index[n2]: fitness_index[n2]=fitness1
    print("Fitness",max(fitness_index))
    #copyover
    if fitness1>fitness2:
        gene_pop[n2]=copy.deepcopy(gene_pop[n1])
    elif fitness2>fitness1:
        gene_pop[n1]=copy.deepcopy(gene_pop[n2])
    
    #cv2.imshow("flow",op) #show grey scale
    keyCode = cv2.waitKey(30) & 0xFF
    # Stop the program on the ESC key
    if keyCode == 27:
                break
    

camera.release()
#{"mode":"full","isActive":false}
{"mode":"full","isActive":false}
for i in range(pop_size): #save all the genes
      np.save(str(i),gene_pop[i])

np.save(str(i),np.array(fitness_index)) #save fitnesses

plt.show()
