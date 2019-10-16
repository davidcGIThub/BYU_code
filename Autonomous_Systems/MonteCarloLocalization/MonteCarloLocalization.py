#MonteCarloLocalization
import numpy as np 

class mcl:
    
    def __init__(self, dt = 0.1, 
                 alpha = np.array([0.1,0.01,0.01,0.1]),
                 sig_r = 0.1,
                 sig_ph = 0.05,):
        self.dt = dt #time step
        self.alpha = alpha #control noise characteristics
        self.sig_r = sig_r #sensor noise (range)
        self.sig_ph = sig_ph #sensor noise (bearing)

    def MonteCarlo_Localization(self, ki, u, z , m):
        #sample the motion model
        ki_x = ki[0,:]
        ki_y = ki[1,:]
        ki_th = ki[2,:]
        v_hat = u[0] + (self.alpha1 * u[0]**2 + self.alpha2 * u[1]**2) * np.random.randn()
        w_hat = u[1] + (self.alpha3 * u[0]**2 + self.alpha4 * u[1]**2) * np.random.randn()
        ki_bar_x = ki_x - v_hat/w_hat * np.sin(ki_th)  + v_hat/w_hat*np.sin(ki_th+w_hat*self.dt)
        ki_bar_y = ki_y + v_hat/w_hat * np.cos(ki_th) -  v_hat/w_hat*np.cos(ki_th+w_hat*self.dt)
        ki_bar_th = ki_th + w_hat*self.dt
        ki_bar = np.array([ki_bar_x, ki_bar_y, ki_bar_th])
        
        