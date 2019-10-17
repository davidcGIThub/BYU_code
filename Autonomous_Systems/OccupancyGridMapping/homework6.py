import numpy as np
import cv2
from RobotPath import RobotPath
from scipy.io import loadmat
from OccupancyGridMapping import OGM

matdata = loadmat('BYU_code/Autonomous_Systems/OccupancyGridMapping/state_meas_data.mat')
X = matdata['X']
z = matdata['z']
x_data = X[0]
y_data = X[1]
th_data = X[2]
alpha = 1.0
beta = 5.0/180*np.pi
z_max = 150.0
pm_hit = 0.7

width = 100
height = 100
FPS = 1000
resize_factor = 10

rp  = RobotPath(x_data,y_data,th_data)
ogm = OGM(alpha, beta, z_max,pm_hit, height, width)
last_image = np.array([[[]]])

for i in range(np.size(X,1)):
    #frame = np.random.randint(0, 256, 
    #                          (height, width), 
    #                          dtype=np.uint8)
    frame = ogm.occupancy_Grid_Mapping(X[:,i],z[:,:,i])
    #frame = ogm.getMap()
    #frame = np.zeros([height,width],dtype=np.uint8)
    enlarged_frame = np.repeat(np.repeat(frame,resize_factor,axis=0),resize_factor,axis=1)
    pts = np.array([rp.getPoints(i,resize_factor)], dtype = np.int32)
    cv2.fillPoly(enlarged_frame,pts,255)
    flipped = cv2.flip(enlarged_frame,0)
    cv2.imshow('Image',flipped)
    cv2.waitKey(int(1.0/FPS*1000))
    if i == np.size(X,1)-1:
        last_image = flipped

cv2.imshow('Image',last_image)
cv2.waitKey()