import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

width = 2
height = 2
FPS = 24
seconds = 2


fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./noise.avi', fourcc, float(FPS), (width, height),0)
#path = r'random-small-images43.png'
#image = cv2.imread(path)
#grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.array([[0,255],[255,0]])
image = image.astype(np.uint8)
print(image)
cv2.imshow('Image',image)
cv2.waitKey(3000)

for _ in range(FPS*seconds):
    frame = np.random.randint(0, 256, 
                              (height, width, 1), 
                              dtype=np.uint8)
    cv2.imshow('Image',frame)
    video.write(frame)

video.release()