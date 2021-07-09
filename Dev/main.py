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
import torch
import random
import time

#from adafruit_motorkit import MotorKit
import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

camera=None

###########
#Define required functions for GA
###########

def fitness(self):
    pass

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

###########
#Define hardware interaction functions and classes
###########

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
        x = torch.from_numpy(x).unsqueeze(0)
        x=torch.mm(x, self.weights.T) #first layer
        return torch.mm(x,self.weights2.T) + self.bias #secon layer
        
    def get_action(self, x):
        #print(self.forward(x))
        a=[]
        for i in list(self.forward(x)[0]):
            if i > 0: #foward
                a.append(1)
            else: #backward
                a.append(0)
        return a
        
###########
#Define image interaction functions
###########
def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=820,
    display_height=616,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def getImage(): #return the image
    if cv2.getWindowProperty("flow", 0) >= 0:
        _,frame=camera.read()
        #add in any preprocessing here
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        scale_percent = 50 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame=cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
        return copy.deepcopy(frame)

def getOpticalFlow(im1,im2): #get the optical flow from previous, current 
    flow = cv2.calcOpticalFlowFarneback(im1,im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX) #horizontal flow
    horz = horz.astype('uint8')
    return horz
###########
#Define needed variables
###########
Generations=500
Rate=0.2
initial_population=[]

# Raspberry Pi pin configuration:
RST = 24
# Raspberry Pi pin configuration:
RST = 24
# Note the following are only used with SPI:
DC = 23
SPI_PORT = 0
SPI_DEVICE = 0

# 128x32 display with hardware I2C:
disp1 = Adafruit_SSD1306.SSD1306_128_32(i2c_bus=8,rst=RST)
disp2 = Adafruit_SSD1306.SSD1306_128_32(i2c_bus=1,rst=RST)

#define camera
camera=cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
while camera.isOpened()==False: pass #wait for it to load


#define motors
#kit = MotorKit()

###########
#Main loop
###########

# Initialize library.
disp1.begin()
disp2.begin()
# Clear display.
disp1.clear()
disp2.clear()
disp1.display()
disp2.display()

# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
width = disp1.width
height = disp1.height
image = Image.new('1', (width, height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a black filled box to clear the image.
draw.rectangle((0,0,width,height), outline=0, fill=0)

# Draw some shapes.
# First define some constants to allow easy resizing of shapes.
padding = 2
shape_width = 20
top = padding
bottom = height-padding
# Move left to right keeping track of the current x position for drawing shapes.
x = padding


# Load default font.
font = ImageFont.load_default()


# Write two lines of text.
draw.text((x, top),    'NANOSAUR',  font=font, fill=255)
draw.text((x, top+20), 'PANORAMASAURUS NX!', font=font, fill=255)

# Display image.
disp1.image(image)
disp1.display()

# Display image.
disp2.image(image)
disp2.display()

_,frame=camera.read()



pixels=frame.shape[0:2]
print(pixels)
cv2.namedWindow("norm", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("flow", cv2.WINDOW_AUTOSIZE)

prvs=getImage()

agent=Agent(pixels[0]*pixels[1],100,2) #h*w inputs for pixels
pop_size=10

gene_pop = []

#for i in range(pop_size): #vary from 10 to 20 depending on purpose of robot
#  gene_pop.append(np.random.normal(0, 0.1, (agent.num_genes)))#create

Generations=500
print("Begin")

for gen in range(Generations):
    #perform Reinforcement learning 
    current=getImage()
    print("Gen",gen,current.shape)
    cv2.imshow("norm",current) #show grey scale
    op=getOpticalFlow(prvs,current) #get the optical flow image for input layer
    #op_grey = cv2.cvtColor(op,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("flow",op) #show grey scale
    keyCode = cv2.waitKey(30) & 0xFF
    # Stop the program on the ESC key
    if keyCode == 27:
                break
    prvs = copy.deepcopy(current)

camera.release()
