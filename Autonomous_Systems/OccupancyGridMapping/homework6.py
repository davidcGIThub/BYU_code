import numpy as np
import cv2
from RobotPath import RobotPath


width = 100
height = 100
FPS = 24
seconds = 2

rp  = RobotPath()

for _ in range(FPS*seconds):
    frame = np.random.randint(0, 256, 
                              (height, width), 
                              dtype=np.uint8)
    enlarged_frame = np.repeat(np.repeat(frame,10,axis=0),10,axis=1)
    pts = np.array([[[500,500],[600,500],[550,600]]], dtype = np.int32)
    cv2.fillPoly(enlarged_frame,pts,0)
    cv2.imshow('Image',enlarged_frame)
    cv2.waitKey(int(1/FPS*1000)) 