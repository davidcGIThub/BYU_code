import numpy as np
import cv2
from RobotPath import RobotPath
from scipy.io import loadmat
import matplotlib.pyplot as plt

matdata = loadmat('state_meas_data.mat')
X = matdata['X']
x_data = X[0]
y_data = X[1]
th_data = X[2]

width = 100
height = 100
FPS = 1000
resize_factor = 10

rp  = RobotPath(x_data,y_data,th_data)

for i in range(np.size(X,1)):
    #frame = np.random.randint(0, 256, 
    #                          (height, width), 
    #                          dtype=np.uint8)
    frame = np.zeros([height,width],dtype=np.uint8)
    enlarged_frame = np.repeat(np.repeat(frame,resize_factor,axis=0),resize_factor,axis=1)
    pts = np.array([rp.getPoints(i,resize_factor)], dtype = np.int32)
    cv2.fillPoly(enlarged_frame,pts,255)
    flipped = cv2.flip(enlarged_frame,0)
    cv2.imshow('Image',flipped)
    cv2.waitKey(int(1.0/FPS*1000)) 