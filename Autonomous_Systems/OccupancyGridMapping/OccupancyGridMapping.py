#occupancyGridMapping
import numpy as np


#p(m_i) = 0.6 to 0.7 if hit detected
#p(m_i) = 0.3 to 0.4 if no hit
class ogm:
    
    def __init__(self, 
                alpha = 1, #m
                beta = 5/180*np.pi, #rad
                z_max = 150): #m

        self.alpha = alpha
        self.beta = beta
        self.z_max = z_max
        self.lt_map = 

    def inverse_range_sensor_model(self,m,x,z):
        mx = m[0]
        my = m[1]
        r = np.sqrt(mx - )

    def occupancy_Grid_Mapping(self,x,z):
