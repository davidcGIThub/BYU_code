#velocity motion model
import numpy as np 

class RobotPath:

    def __init__(self, 
                 x_path,
                 y_path,
                 th_path):
        self.x_path = x_path
        self.y_path = y_path
        self.th_path = th_path

    def getState(self,index):
        x = self.x[index]
        y = self.y[index]
        th = self.th[index]
        return np.array([x,y,th])

    def getPoints(self,index):
        x = self.x[index]
        y = self.y[index]
        th = self.th[index]
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        xy = np.array([[-1, 1, -1],
                       [.5, 0, -0.5]])
        xy = np.dot(R,xy)
        xy = xy + np.array([[x],[y]])
        return np.transpose(xy)
