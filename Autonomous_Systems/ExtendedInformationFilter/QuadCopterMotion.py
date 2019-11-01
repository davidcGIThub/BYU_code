#velocity motion model
import numpy as np 

class QuadCopterMotion:

    def __init__(self, 
                 x = -5.0, 
                 y = 0.0, 
                 theta = np.pi/2.0, 
                 sig_v =  0.15, #m/s
                 sig_w = 0.1, #rad/s
                 dt = 0.1):
        self.x = x
        self.y = y
        self.theta = theta
        self.sig_v = sig_v
        self.sig_w = sig_w
        self.dt = dt
    
    def setState(self,x,y,theta):
        self.x = x
        self.y = y
        self.theta = theta

    def vel_motion_model(self,u,true_motion = False):
        v = u[0]
        w = u[1]
        if true_motion == False:
            v_hat = v + sig_v * np.random.randn()
            w_hat = w + sig_w * np.random.randn()
        else:
            v_hat = v
            w_hat = w
        self.x = self.x + v_hat*np.cos(self.theta)*self.dt
        self.y = self.y + v_hat*np.sin(self.theta)*self.dt
        self.theta = self.theta + w_hat*self.dt

    def getState(self):
        return np.array([self.x,self.y,self.theta])

    def getPoints(self):
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                      [np.sin(self.theta), np.cos(self.theta)]])
        xy = np.array([[-1, 1, -1],
                       [.5, 0, -0.5]])
        xy = np.dot(R,xy)
        xy = xy + np.array([[self.x],[self.y]])
        return np.transpose(xy)

    #plt.fill([x1,x2,x3],[y1,y2,y3])