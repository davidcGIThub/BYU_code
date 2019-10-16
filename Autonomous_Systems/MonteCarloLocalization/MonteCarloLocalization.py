<<<<<<< HEAD
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
        
        
=======
#Monte Carlo
import numpy as np 



class MCL:

    def __init__(self,dt = 0.1,
                      alpha = np.array([0.1,0.01,0.01,0.1]),
                      sig_r = 0.1,
                      sig_ph = 0.05,
                      M = 1000):
        self.dt = dt
        self.alpha = alpha #control noise characteristics
        self.sig_r = sig_r #sensor noise (range)
        self.sig_ph = sig_ph #sensor noise (bearing)
        self.M = M # number of particles

    def prob_normal_distribution(self, a, std):
        return np.exp(-(a**2)/(2*std**2)) / np.sqrt(2*np.pi*std**2)

    def low_variance_sampler(self, ki_bar, w):
        r = np.random.uniform(0,1.0/float(self.M))
        c = w[0]
        i = 0
        ki = ki_bar*0
        for k in range(1,self.M+1):
            U = r + (k-1)/float(self.M)
            while U > c:
                i = i+1
                c = c + w[i]
            ki[:,k-1] = ki_bar[:,i]
        return ki

    def MCL_Localization(self, ki_past, u, z, m):
        #sample the motion model
        if np.size(m,0) != np.size(z,1):
            print("error: range and bearing measurements do not match landmark count")
        v_hat = u[0] + (self.alpha[0] * u[0]**2 + self.alpha[1] * u[1]**2) * np.random.randn(self.M)
        w_hat = u[1] + (self.alpha[2] * u[0]**2 + self.alpha[3] * u[1]**2) * np.random.randn(self.M)
        ki_bar_x = ki_past[0,:] - v_hat/w_hat * np.sin(ki_past[2,:])  + v_hat/w_hat*np.sin(ki_past[2,:]+w_hat*self.dt)
        ki_bar_y = ki_past[1,:] + v_hat/w_hat * np.cos(ki_past[2,:]) -  v_hat/w_hat*np.cos(ki_past[2,:]+w_hat*self.dt)
        ki_bar_th = ki_past[2,:] + w_hat*self.dt
        ki_bar = np.array([ki_bar_x, ki_bar_y, ki_bar_th])
        #measurement model probability
        w = np.zeros(self.M) + 1.0
        for i in range(0,np.size(m,0)):
            Range = z[0,i]
            Bearing = z[1,i]
            Range_ki = np.sqrt((m[i,0] - ki_bar_x)**2 + (m[i,1] - ki_bar_y)**2)
            Bearing_ki = np.arctan2(m[i,1] - ki_bar_y, m[i,0] - ki_bar_x) - ki_bar_th
            prob_R = self.prob_normal_distribution(Range_ki - Range, self.sig_r)
            prob_B = self.prob_normal_distribution(Bearing_ki - Bearing, self.sig_ph)
            w = w * prob_R * prob_B
        #Resampling
        w = w/np.sum(w)
        ki = self.low_variance_sampler(ki_bar, w)
        mu = np.mean(ki,1)
        return ki, mu
>>>>>>> 73451215196ddeacb1e181b0702f6ee3e13ad5ad
