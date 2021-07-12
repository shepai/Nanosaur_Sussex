import cv2
import numpy as np
import time

# deprecated, checks if point in the sphere is in our output
def isInROI(x,y,R1,R2,Cx,Cy):
    isInOuter = False
    isInInner = False
    xv = x-Cx
    yv = y-Cy
    rt = (xv*xv)+(yv*yv)
    if( rt < R2*R2 ):
        isInOuter = True
        if( rt < R1*R1 ):
            isInInner = True
    return isInOuter and not isInInner
# build the mapping
def buildMap(Ws,Hs,Wd,Hd,R1,R2,Cx,Cy):
    map_x = np.zeros((Hd,Wd),np.float32)
    map_y = np.zeros((Hd,Wd),np.float32)
    for y in range(0,int(Hd-1)):
        for x in range(0,int(Wd-1)):
            r = (float(y)/float(Hd))*(R2-R1)+R1
            theta = (float(x)/float(Wd))*2.0*np.pi
            xS = Cx+r*np.sin(theta)
            yS = Cy+r*np.cos(theta)
            map_x.itemset((y,x),int(xS))
            map_y.itemset((y,x),int(yS))
        
    return map_x, map_y
# do the unwarping 
def unwarp(img,xmap,ymap):
    output = cv2.remap(img,xmap,ymap,cv2.INTER_LINEAR)
    return output


disp = (820,616)
vals = []
last = (0,0)
# Load the video from the rpi
#vc = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
# Sometimes there is crud at the begining, buffer it out
for i in range(1,10):
    img = cv2.imread('/home/jetsonnx/Documents/photoDat/image'+str(i)+'.png')
    #cv2.imwrite('/home/jetsonnx/Documents/photoDat/image'+str(i)+'.png',img)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 0 = xc yc
# 1 = r1
# 2 = r2
# center of the "donut"    
Cx = 410
Cy = 308
# Inner donut radius
R1x = 461
R1y = 336
R1 = int(R1x-Cx)
# outer donut radius
R2x = 153
R2y = 342
R2 = int(R2x-Cx)
# our input and output image siZes
Wd = abs(int(2.0*((R2+R1)/2)*np.pi))
Hd = abs(int(R2-R1))
Ws = img.shape[1]
Hs = img.shape[0]
# build the pixel map, this could be sped up
print ("BUILDING MAP!")
xmap,ymap = buildMap(Ws,Hs,Wd,Hd,R1,R2,Cx,Cy)
print ("MAP DONE!")
# do an unwarping and show it to us
result = unwarp(img,xmap,ymap)
result=cv2.flip(result,0)
cv2.imwrite('/home/jetsonnx/Documents/image.png',result)


