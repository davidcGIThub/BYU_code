#occupancyGridMapping
import numpy as np


#p(m_i) = 0.6 to 0.7 if hit detected
#p(m_i) = 0.3 to 0.4 if no hit
class OGM:
    
    def __init__(self, 
                alpha = 1.0,            #m
                beta = 5.0/180*np.pi,   #rad
                z_max = 150.0,          #m
                pm_hit = 0.7,           #probability of mass if hit detected
                height = 100,
                width = 100):

        self.alpha = alpha
        self.beta = beta
        self.z_max = z_max
        self.pm_hit = pm_hit
        self.pm_miss = 1.0 - pm_hit
        self.locc = np.log(self.pm_hit/(1.0-self.pm_hit))
        self.lfre = np.log(self.pm_miss/(1.0-self.pm_miss))
        self.lo = np.log(1)
        self.height = height
        self.width = width
        self.lt_map = np.zeros([height,width]) + self.lo
        self.map = np.zeros([height,width]) + 1 - 1/(1+np.exp(self.lo))

    def inverse_range_sensor_model(self,m,X,z):
        mx = m[0]
        my = m[1]
        x = X[0]
        y = X[1]
        th = X[2]

        r = np.sqrt((mx - x)**2 + (my-y)**2)   #range
        phi = np.arctan2(my-y,mx-x) - th       #bearing
        k = np.argmin(np.abs(phi - z[1,:]))
        r_k = z[0,k]
        phi_k = z[1,k]
        if r > np.min([self.z_max , r_k + self.alpha/2]) or np.abs(phi-phi_k) > self.beta/2:
            l = self.lo
        elif r_k < self.z_max and np.abs(r - r_k) < self.alpha/2.0:
            l = self.locc
        elif r <= r_k:
            l = self.lfre
        else:
            l = self.lo
        return l

    def occupancy_Grid_Mapping(self,X,z):
        for i in range(0,self.height):
            for j in range(0,self.width):
                m = np.array([j+1,i+1])
                self.lt_map[i,j] = self.lt_map[i,j] + self.inverse_range_sensor_model(m,X,z)  - self.lo
                self.map[i,j] = 1-1/(1+np.exp(self.lt_map[i,j]))
        return self.map
    
    #def getMap(self):
    #    return 1-1/(1+np.exp(self.lt_map))