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
        v_hat = u[0] + (self.alpha[0] * u[0]**2 + self.alpha[1] * u[1]**2) * np.random.randn(self.M) #estimated velocity
        w_hat = u[1] + (self.alpha[2] * u[0]**2 + self.alpha[3] * u[1]**2) * np.random.randn(self.M) #estimated angular velocity
        ki_bar_x = ki_past[0,:] - v_hat/w_hat * np.sin(ki_past[2,:])  + v_hat/w_hat*np.sin(ki_past[2,:]+w_hat*self.dt) #propogate particles (x pos)
        ki_bar_y = ki_past[1,:] + v_hat/w_hat * np.cos(ki_past[2,:]) -  v_hat/w_hat*np.cos(ki_past[2,:]+w_hat*self.dt) #propogate particles (y pos)
        ki_bar_th = ki_past[2,:] + w_hat*self.dt #propogate particles (theta angle)
        ki_bar = np.array([ki_bar_x, ki_bar_y, ki_bar_th]) 
        #measurement model probability
        w = np.zeros(self.M) + 1.0
        for i in range(0,np.size(m,0)): #loop through each landmark
            Range = z[0,i] # landmark range
            Bearing = z[1,i] #landmark bearing
            Range_ki = np.sqrt((m[i,0] - ki_bar_x)**2 + (m[i,1] - ki_bar_y)**2) #range of the particles
            Bearing_ki = np.arctan2(m[i,1] - ki_bar_y, m[i,0] - ki_bar_x) - ki_bar_th # bearing of particles
            prob_R = self.prob_normal_distribution(Range_ki - Range, self.sig_r) #range probability
            prob_B = self.prob_normal_distribution(Bearing_ki - Bearing, self.sig_ph) #bearing probability
            w = w * prob_R * prob_B #weights
        #Resampling
        w = w/np.sum(w) #normalize the weights
        ki = self.low_variance_sampler(ki_bar, w) #particles
        unique = np.size(np.unique(ki,axis=1))
        P = np.cov(ki_bar)
        n = 3 # number of states
        if 1.0*unique/self.M < 0.5:
            Q = P/((1.0*self.M*unique)**(1.0/n))
            ki = ki + np.dot(Q,np.random.randn(n,self.M))
        mu = np.mean(ki,1) #estimated pose (average of pose of particles)
        return ki, mu, P
