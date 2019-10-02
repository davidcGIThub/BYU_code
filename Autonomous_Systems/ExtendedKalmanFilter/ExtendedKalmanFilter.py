#ExtendedKalmanFilter
import numpy as np 

class EKF:

    def __init__(self, dt = 0.1, 
                       alpha1 = 0.1, 
                       alpha2 = 0.01, 
                       alpha3 = 0.01, 
                       alpha4 = 0.1,
                       sig_r = 0.1,
                       sig_ph = 0.05):
        self.dt = dt;
        self.alpha1 = alpha1;
        self.alpha2 = alpha2;
        self.alpha3 = alpha3;
        self.alpha4 = alpha4;
        self.sig_r = sig_r;
        self.sig_ph = sig_ph;


    def EKF_Localization(self, mu, sig, u, z, c, m):
        #mu in the last time step 
        mu_x = mu[0];
        mu_y = mu[1];
        mu_th = mu[2];
        #control input
        vc = u[0];
        wc = u[1];
        #use prior theta to predict current state
        theta = mu_th;
        #jacobian of g(u(t),x(t-1))
        G = np.identity(3);
        G[0][2] = -vc/wc*np.cos(theta) + vc/wc*np.cos(theta+wc*self.dt);
        G[1][2] = -vc/wc*np.sin(theta) + vc/wc*np.sin(theta+wc*self.dt);
        #Jacobian to map noise from control space to state space
        V = np.zeros(3,2);
        V[0][0] = ( -np.sin(theta) + np.sin(theta + wc*self.dt) ) / wc;
        V[0][1] = ( vc * (np.sin(theta) - np.sin(theta + wc*self.dt)) ) / wc**2 + ( vc*np.cos(theta + wc*self.dt)*self.dt ) / wc; 
        V[1][0] = ( np.cos(theta) - np.cos(theta + wc*self.dt) ) / wc;
        V[1][1] = - ( vc * (np.cos(theta) - np.cos(theta+wc*self.dt)) ) / wc**2 + ( vc * np.sin(theta + wc*self.dt)*self.dt ) / wc;
        V[2][1] = self.dt;
        #control noise covariance
        M = np.zeros(2,2);
        M[0][0] = self.alpha1*vc**2 + self.alpha2*wc**2;
        M[1][1] = self.alpha3*vc**2 + self.alpha4*wc**2;
        #state estimate - prediction step
        mu_x = mu_x - vc*np.sin(theta)/wc + vc*np.sin(theta+wc*self.dt)/wc;
        mu_y = mu_y + vc*np.cos(theta)/wc - vc*np.cos(theta+wc*self.dt)/wc;
        mu_th = mu_th + wc*self.dt;
        #state covariance - prediction step
        Sig = np.dot( G ,np.dot(Sig,np.transpose(G)) ) + np.dot( V, np.dot(M,np.transpose(V)) );
        #Uncertainty due to measurement noise
        Q = np.zeros(2,2);
        Q[0][0] = self.sig_r**2;
        Q[1][1] = self.sig_ph**2;





