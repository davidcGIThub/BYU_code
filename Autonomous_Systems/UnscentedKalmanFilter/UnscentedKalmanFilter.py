#Unscented Kalman Filter
import numpy as np 
from LandmarkModel import LandmarkModel as lmm 
from scipy.linalg import block_diag


class UKF:

    def __init__(self,dt = 0.1,
                      alpha = np.array([0.1,0.01,0.01,0.1]),
                      sig_r = 0.1,
                      sig_ph = 0.05,
                      alfa = 0.5,
                      kappa = 3.0,
                      beta = 2):
        self.dt = dt;
        self.alpha = alpha; #control noise characteristics
        self.sig_r = sig_r; #sensor noise (range)
        self.sig_ph = sig_ph; #sensor noise (bearing)
        self.alfa = alfa; #scaling parameters
        self.kappa = kappa; #scaling parameters
        self.beta = beta; #Guassian distribution 

    def UKF_Localization(self, mu, Sig, u, z, m):
        len_mu = np.size(mu);
        len_u = np.size(u);
        len_z = np.size(z);
        n = len_mu + len_u + len_z; #length of augmented state vector
        v = u[0]; #velocity command
        w = u[1]; #angular velocity command
        #Generate Augmented Mean and Covariance
        M = np.array([[self.alpha[0]*v**2 + self.alpha[1]*w**2, 0],
                      [0 , self.alpha[2]*v**2 + self.alpha[3]*w**2]]); #control noise covariance matrix
        Q = np.array([[self.sig_r**2 , 0],
                      [0 , self.sig_ph**2]]); #measurement noise covariance matrix
        mu_a = np.zeros(n);
        mu_a[0:len_mu] = mu[0:len_mu]; #mean of augmented state
        Sig_a = block_diag(Sig,M,Q); #covariance of Augmented state
        #Generate Sigma Points
        lamda = (self.alfa**2)*(n+self.kappa) - n;
        gamma = np.sqrt(n+lamda);
        ki_a = np.hstack( mu_a[:,None] , mu_a[:,None]+gamma*np.sqrt(Sig_a) , mu_a[:,None]-gamma*np.sqrt(Sig_a));
        #Pass Sigma points through motion model and compute Guassian statistics
        v_i = v + Sig_a[3,:];
        w_i = w + Sig_a[4,:];
        theta_i = Sig_a[2,:];
        x_row = -v_i/w_i*np.sin(theta_i) + v_i/w_i*np.sin(theta_i+w_i*self.dt);
        y_row = v_i/w_i*np.cos(theta_i) - v_i/w_i*np.cos(theta_i+w_i*self.dt);
        theta_row = w_i*self.dt;
        Sig_x = Sig_a[0:3,:] + np.array([x_row, y_row, theta_row]);

        #