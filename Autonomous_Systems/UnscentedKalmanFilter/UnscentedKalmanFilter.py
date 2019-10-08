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
        len_z = np.size(z,0);
        n = len_mu + len_u + len_z; #length of augmented state vector
        v = u[0]; #velocity command
        w = u[1]; #angular velocity command
        #Generate Augmented Mean and Covariance
        M = np.array([[self.alpha[0]*v**2 + self.alpha[1]*w**2, 0],
                      [0 , self.alpha[2]*v**2 + self.alpha[3]*w**2]]); #control noise covariance matrix
        print('M 2x2' , np.shape(M));
        Q = np.array([[self.sig_r**2 , 0],
                      [0 , self.sig_ph**2]]); #measurement noise covariance matrix
        print('Q 2x2' , np.shape(Q));
        mu_a = np.zeros(n);
        mu_a[0:len_mu] = mu[0:len_mu]; #mean of augmented state
        print('mu_a 7x1' , np.shape(mu_a));
        Sig_a = block_diag(Sig,M,Q); #covariance of Augmented state
        print('Sig_a 7x7' , np.shape(Sig_a));
        #Generate Sigma Points
        lamda = (self.alfa**2)*(n+self.kappa) - n;
        gamma = np.sqrt(n+lamda);
        ki_a = np.hstack( (mu_a[:,None] , mu_a[:,None]+gamma*np.linalg.cholesky(Sig_a) , mu_a[:,None]-gamma*np.linalg.cholesky(Sig_a)) );
        print('ki_a 7x15' , np.shape(ki_a));
        #Pass Sigma points through motion model and compute Guassian statistics
        v_i = v + ki_a[3,:];
        w_i = w + ki_a[4,:];
        theta_i = ki_a[2,:];
        x_row = -v_i/w_i*np.sin(theta_i) + v_i/w_i*np.sin(theta_i+w_i*self.dt);
        y_row = v_i/w_i*np.cos(theta_i) - v_i/w_i*np.cos(theta_i+w_i*self.dt);
        theta_row = w_i*self.dt;
        ki_x = ki_a[0:3,:] + np.array([x_row, y_row, theta_row]);
        print('ki_x 3x15' , np.shape(ki_x))
        w_m = np.arange(15);
        w_c = np.arange(15);
        w_m[0] = lamda /(n+lamda);
        w_c[0] = lamda / (n+lamda) + (1 - self.alfa**2 + self.beta);
        arr = np.arange(14) + 1;
        w_m[1:15] = 1 / (2*(arr+lamda));
        w_c[1:15] = w_m[1:15];
        mu_est = np.dot(ki_x,w_m[:,None]);
        print('mu_est 3x1' , np.shape(mu_est));
        Sig_est = np.dot(w_c*(ki_x-mu_est) , np.transpose(ki_x-mu_est));
        print('Sig_est 3x3' , np.shape(Sig_est));
        #Predict observations at sigma points and comput Gaussian statistics
        x_mark = m[0,0];
        y_mark = m[0,1];
        Z = np.array([ np.sqrt((x_mark - ki_x[0,:])**2 + (y_mark - ki_x[1,:])**2) + ki_a[5,:] , 
                      np.arctan2(y_mark-ki_x[1,:] , x_mark-ki_x[0,:]) - ki_x[2,:] + ki_a[6,:]]);
        print('Z 2x15', np.shape(Z))
        z_est = np.dot(Z,w_m[:,None]);
        print('z_est 2x1', np.shape(z_est));
        S = np.dot(w_c*(Z-z_est) , np.transpose(Z-z_est));
        print('S 2x2', np.shape(S));
        Sig_cross = np.dot(w_c*(ki_x-mu_est) , np.transpose(Z-z_est));
        print("Sig_cros 3x2" , np.shape(Sig_cross))
        #Update mean and covariance
        K = np.dot(Sig_cross , np.linalg.inv(S));
        print('K 3x2' , np.shape(K) )
        mu_est = mu_est + np.dot(K , z[:,0,None] - z_est);
        print('mu_est 3x1' , np.shape(mu_est));
        Sig_est = Sig_est - np.dot( K , np.dot(S,np.transpose(K)) );
        print('Sig_est 3x3' , np.shape(Sig_est));
        return mu_est.flatten(), Sig_est
    