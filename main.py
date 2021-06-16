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

import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

camera=cv2.VideoCapture(0) #gather the camera from the input 

###########
#Define required functions for GA
###########

def fitness(self):
    pass

def mutate(self,genotype):
    pass

def crossover(self,genotype1,genotype2):
    pass

###########
#Define hardware interaction functions and classes
###########
class motor:
    def __init__(self,pin1,pin2):
        self.pin1=pin1
        self.pin2=pin2
    def start(self,direction): #accept direction as boolean
        if direction:
            pass #forward
        else:
            pass #backward
    def stop(self):
        pass

def is_obstructed():
    #check whether the bot is obstructed
    return False

def return_to_start(moves):
    #reverse moves
    #return to start
    pass

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
Generations=500
Rate=0.2
initial_population=[]

# Raspberry Pi pin configuration:
RST = 24
# Note the following are only used with SPI:
DC = 23
SPI_PORT = 0
SPI_DEVICE = 0

# 128x32 display with hardware I2C:
disp1 = Adafruit_SSD1306.SSD1306_128_32(i2c_bus=1,rst=RST)

# 128x32 display with hardware I2C:
disp2 = Adafruit_SSD1306.SSD1306_128_32(i2c_bus=0,rst=RST)


#define motors
motor1=motor(1,2)
motor2=motor(3,4)


###########
#Main loop
###########

# Initialize library.
disp1.begin()
disp2.begin()
# Clear display.
disp.clear()
disp.display()


for gen in range(Generations):
    #perform Reinforcement learning 
    pass
