#velocity motion model
import numpy as np 

class RobotPath:

    def __init__(self, 
                 x_path,
                 y_path,
                 th_path,
                 bot_len = 1.0):
        self.x_path = x_path
        self.y_path = y_path
        self.th_path = th_path
        self.bot_len = bot_len

    def getState(self,index):
        x = self.x_path[index]
        y = self.y_path[index]
        th = self.th_path[index]
        return np.array([x,y,th])

    def getPoints(self,index,size_factor):
        x = self.x_path[index]
        y = self.y_path[index]
        th = self.th_path[index]
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        xy = np.array([[-self.bot_len, self.bot_len, -self.bot_len],
                       [self.bot_len/2, 0, -self.bot_len/2]]) * size_factor
        xy = np.dot(R,xy)
        xy = xy + np.array([[x],[y]])*size_factor
        return np.transpose(xy)
