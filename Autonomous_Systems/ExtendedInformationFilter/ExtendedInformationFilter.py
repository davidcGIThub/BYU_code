#ExtendedInformationFilter
import numpy as np 

class EIF:

    def __init__(self, dt = 0.1, 
                       sig_v = 0.15, #m/s
                       sig_w = 0.1, #rad/s
                       sig_r = 0.1, #m
                       sig_ph = 0.05,
                       landmarks = np.array([[6,4],[-7,8],[12,-8],[-2,0],[-10,2],[13,7]])): #rad
        self.dt = dt
        self.sig_v = 0.15
        self.sig_w = 0.1
        self.sig_r = sig_r
        self.sig_ph = sig_ph
        self.landmarks = landmarks

    def EIF_localization(self,xi,omega,mu,sigma,u,z):
        v = u[0]
        v_hat = v + self.sig_v*np.random.randn()
        w = u[1]
        w_hat = w + self.sig_w*np.random.randn()
        x = mu[0]
        y = mu[1]
        theta = mu[2]
        G = np.array([[1,        0,       -v*np.sin(theta)*self.dt],
                      [0,        1,        v*np.cos(theta)*self.dt],
                      [0,        0,        1]])
        M = np.array([[self.sig_v*np.random.randn(),            0],
                      [0,            self.sig_w*np.random.randn()]])
        V = np.array([[np.cos(theta)*self.dtdt ,        0],
                      [np.sin(theta)*self.dtdt ,        0],
                      [0                       , self.dt]])
        omega_bar = np.linalg.inv( np.dot(G , np.dot(sigma,np.transpose(G))) )
        mu_bar = mu + np.array([v_hat*np.cos(theta)*self.dt, v_hat*np.sin(theta)*self.dt, w_hat*dt])
        xi_bar = np.dot(omega_bar,mu_bar[:,None])
        Q = np.array([[self.sig_r**2, 0],
                      [0,   self.sig_phi**2]])
        Qinv = np.linalg.inv(Q)
        #omega_bar = omega_bar + 
        num_landmarks = np.size(self.landmarks,0)
        for i in range(0,num_landmarks):
            landmark = self.landmarks[i,:]
            q = (landmark[0] - mu_bar[0])**2 + (landmark[1] - mu_bar[1])**2
            b = np.arctan2(landmark[1] - mu_bar[1], landmark[0] - mu_bar[0]) - mu_bar[2]
            h = np.array([[np.sqrt(q)] , [b]])
            z_t = (z[i,:])[:,None]
            H = np.array([[-(landmark[0] - mu_bar[0])/np.sqrt(q)    ,   -(landmark[1] - mu_bar[1])/np.sqrt(q)   ,   0],
                          [(landmark[1] - mu_bar[1])/q             ,   -(landmark[0] - mu_bar[0])/q            ,  -1.0]])
            omega_bar = omega_bar + np.dot(np.transpose(H) , np.dot(Qinv,H))
            brackets = z_t - h + np.dot(H,mu_bar[:,None]) 
            xi_bar = xi_bar + np.dot( np.transpose(H) , np.dot(Qinv, brackets) )
        sigma_bar = np.linalg.inv(omega_bar)
        mu_bar = np.dot(sigma_bar,xi_bar)
        return xi_bar,omega_bar,mu_bar,sigma_bar