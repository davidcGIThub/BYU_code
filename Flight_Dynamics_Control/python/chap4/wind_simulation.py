"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
import sys
sys.path.append('..')
from tools.transfer_function import transfer_function
import numpy as np

class wind_simulation:
    def __init__(self, Ts, Va0):
        # steady state wind defined in the inertial frame
        self._steady_state = np.array([[0], [0], [0]])
        #medium altitude, light turbulence (alitutde 600 meters)
        sigma_u = 1.5 #m/s
        sigma_v = sigma_u
        sigma_w = 1.5 #m/s
        Lw = 533 #m
        Lu = 533 #m
        Lv = Lu
        a1 = sigma_u*np.sqrt(2*Va0/(np.pi*Lu))
        b1 = Va0/Lu
        a2 = sigma_v*np.sqrt(3*Va0/(np.pi*Lv))
        a3 = a2*Va0/(np.sqrt(3)*Lv)
        b2 = 2*Va0/Lv
        b3 = (Va0/Lv)**2
        a4 = sigma_w*np.sqrt(3*Va0/(np.pi*Lw))
        a5 = a4*Va0 / (np.sqrt(3)*Lw)
        b4 = 2*Va0/Lw
        b5 = (Va0/Lw)**2

        self.u_w = transfer_function(num=np.array([[a1]]),
                                     den=np.array([[1, b1]]),
                                     Ts=Ts)
        self.v_w = transfer_function(num=np.array([[a2, a3]]),
                                     den=np.array([[1, b2, b3]]),
                                     Ts=Ts)
        self.w_w = transfer_function(num=np.array([[a4, a5]]),
                                     den=np.array([[1, b4, b5]]),
                                     Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        #gust = np.array([[0.],[0.],[0.]])
        return np.concatenate(( self._steady_state, gust ))

