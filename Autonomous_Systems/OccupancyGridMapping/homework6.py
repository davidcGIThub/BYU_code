import numpy as np
import cv2
from RobotPath import RobotPath
from scipy.io import loadmat
from OccupancyGridMapping import OGM

matdata = loadmat('BYU_code/Autonomous_Systems/OccupancyGridMapping/state_meas_data.mat')
X = matdata['X']
z = matdata['z']
thk = matdata['thk'][0]
x_data = X[0]
y_data = X[1]
th_data = X[2]
alpha = 1
beta = 2.0/180*np.pi
z_max = 150.0
pm_hit = 0.7

width = 100
height = 100
FPS = 100
resize_factor = 10

rp  = RobotPath(x_data,y_data,th_data)
ogm = OGM(alpha, beta, z_max,pm_hit, height, width,thk)
img_array = []

for i in range(np.size(X,1)):
    frame = ogm.occupancy_Grid_Mapping2(X[:,i],z[:,:,i])
    enlarged_frame = np.repeat(np.repeat(frame,resize_factor,axis=0),resize_factor,axis=1)
    pts = np.array([rp.getPoints(i,resize_factor)], dtype = np.int32)
    cv2.fillPoly(enlarged_frame,pts,255)
    flipped = cv2.flip(enlarged_frame,0)
    cv2.imshow('Image',flipped)
    print(flipped)
    cv2.waitKey(int(1.0/FPS*1000))
    img_array.append(np.reshape((flipped*255).astype(np.uint8) , (height*resize_factor,width*resize_factor,1)))

fourcc = cv2.VideoWriter_fourcc(*'MP42')
video = cv2.VideoWriter('./occ_grid_map_vid.avi',fourcc, float(24), (width*resize_factor, height*resize_factor),0)

for i in range(len(img_array)):
    video.write(img_array[i])
video.release()