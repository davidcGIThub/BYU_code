#occupancyGridMapping
import numpy as np


#p(m_i) = 0.6 to 0.7 if hit detected
#p(m_i) = 0.3 to 0.4 if no hit
class OGM:
    
    def __init__(self, 
                alpha = 1.0,            #m
                beta = 5.0/180*np.pi,   #rad
                r_max = 150.0,          #m
                pm_hit = 0.7,           #probability of mass if hit detected
                height = 100,
                width = 100,
                thk = np.array([-1.5708, -1.2566, -0.9425, -0.6283, -0.3142, 0, 0.3142, 0.6283, 0.9425, 1.2566, 1.5708])):

        self.alpha = alpha
        self.beta = beta
        self.r_max = r_max
        self.pm_hit = pm_hit
        self.pm_miss = 1.0 - pm_hit
        self.locc = np.log(self.pm_hit/(1.0-self.pm_hit))
        self.lfre = np.log(self.pm_miss/(1.0-self.pm_miss))
        self.lo = np.log(1)
        self.height = height
        self.width = width
        self.thk = thk
        self.lt_map = np.zeros([height,width]) + self.lo
        self.my = np.repeat(np.arange(self.height)[:,None], self.width, axis=1)
        self.mx = np.repeat(np.array([np.arange(self.width)]), self.height,axis = 0)

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
        if r > np.min([self.r_max , r_k + self.alpha/2]) or np.abs(phi-self.thk[k]) > self.beta/2:
            l = self.lo
        elif r_k < self.r_max and np.abs(r - r_k) < self.alpha/2.0:
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
        return 1-1/(1+np.exp(self.lt_map))

    def occupancy_Grid_Mapping2(self,X,z):
        x = X[0]
        y = X[1]
        th = X[2]
        r = z[0,:]
        phi = z[1,:]
        r_m = np.sqrt((self.mx - x)**2 + (self.my-y)**2)
        phi_m = np.arctan2(self.my-y,self.mx-x)-th
        #calculate k value
        phi_m_shaped = np.reshape( phi_m ,(self.height,self.width,1))
        phi_shaped = np.repeat( np.repeat(np.array([[phi]]),self.height,0) , self.width, 1)
        k = np.argmin(np.abs(phi_shaped - phi_m_shaped),2)
        r_k = r[k]
        th_k = self.thk[k]
        ######assign l values#######
        #default value self.lo
        l = np.zeros([self.height,self.width]) + self.lo
        #hit conditions
        l[np.abs(r_m - r_k) < self.alpha/2.0] = self.locc
        #miss conditions
        l[r_m < r_k] = self.lfre
        #unknown conditions
        l[r_m >= self.r_max] = self.lo
        l[r_m > r_k+self.alpha/2.0] = self.lo
        l[np.abs(phi_m - th_k) > self.beta/2.0] = self.lo
        #add probability to map
        self.lt_map = self.lt_map + l - self.lo
        return 1-1/(1+np.exp(self.lt_map))