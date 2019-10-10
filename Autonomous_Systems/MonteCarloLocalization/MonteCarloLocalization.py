#Monte Carlo
import numpy as np 
import random



class MCL:

    def __init__(self,dt = 0.1,
                      alpha = np.array([0.1,0.01,0.01,0.1]),
                      sig_r = 0.1,
                      sig_ph = 0.05,):
        self.dt = dt
        self.alpha = alpha #control noise characteristics
        self.sig_r = sig_r #sensor noise (range)
        self.sig_ph = sig_ph #sensor noise (bearing)

    def prob_normal_distribution(self, a, std):
        return np.exp(-(a**2)/(2*std**2)) / np.sqrt(2*np.pi*std**2)

    def low_variance_sampler(self, ki_bar, w):
        M = np.size(w)
        r = 1.0/random.randrange(M)
        c = w[0]
        i = 0
        ki = ki_bar*0
        for k in range(1,M+1):
            U = r + (k-1)/M
            while U > c:
                i = i+1
                c = c + w[i]
            ki[:,k-1] = ki_bar[:,i]
        return ki

    def MCL_Localization(self, ki_past, u, z, m):
        #sample the motion model
        v_hat = u[0] + (self.alpha[0] * u[0]**2 + self.alpha[1] * u[1]**2) * np.random.randn()
        w_hat = [1] + (self.alpha[2] * u[0]**2 + self.alpha[3] * u[1]**2) * np.random.randn()
        ki_bar_x = ki_past[0,:] - v_hat/w_hat * np.sin(ki_past[2,:])  + v_hat/w_hat*np.sin(ki_past[2,:]+w_hat*self.dt)
        ki_bar_y = ki_past[1,:] + v_hat/w_hat * np.cos(ki_past[2,:]) -  v_hat/w_hat*np.cos(ki_past[2,:]+w_hat*self.dt)
        ki_bar_th = ki_past[2,:] + w_hat*self.dt
        ki_bar = np.array([ki_bar_x, ki_bar_y, ki_bar_th])
        #measurement model probability
        Range = z[0,0]
        Bearing = z[1,0]
        Range_ki = np.sqrt((m[0,0] - ki_bar_x)**2 + (m[0,1] - ki_bar_y)**2)
        Bearing_ki = np.arctan2(m[0,1] - ki_bar_y, m[0,0] - ki_bar_x) - ki_bar_th
        prob_R = self.prob_normal_distribution(Range_ki - Range, self.sig_r)
        prob_B = self.prob_normal_distribution(Bearing_ki - Bearing, self.sig_ph)
        w = prob_R * prob_B
        #Resampling
        ki = self.low_variance_sampler(ki_bar, w)
        mu = np.mean(ki,1)
        return ki, mu