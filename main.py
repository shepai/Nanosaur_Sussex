"""

JRA by Dexter R C Shepherd, aged 19
Supervised by Dr James Knight
Mentored by Efstathios Kagioulis

Part of the University of Sussex Junior Research Associate scheme for Deep learning for autonomous navigation on a
small robots. This project uses the Jetson on the NanoSaur chassis.

It makes use of evolutionary strategies to evolve a panoramic image as input and outputs motor control.

This uses a neural network of multiple layers to develop behaviour.

Fitness
Fitness will be measured by whether it bumped into objects (being bad) and whether it found a food source
(being good). The time in which it finds the food source, in consideration to the distance travelled,
will play a part of this fitness.

Mutation
We will use a standard mutation with a guassian, and crossover of genotypes


"""
import cv2
import numpy as np
import copy

camera=cv2.VideoCapture(0) #gather the camera from the input 

###########
#Define required functions for GA
###########


###########
#Define hardware interaction functions
###########


###########
#Define agent class
###########

###########
#Define image interaction functions
###########

def getImage(): #return the image
    _,frame=camera.read()
    #add in any preprocessing here
    return copy.deepcopy(frame)

###########
#Define needed variables
###########


###########
#Main loop
###########
