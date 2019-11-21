#ExtendedKalmanFilter
import numpy as np 
from LandmarkModel import LandmarkModel as lmd

class EKF:

    def __init__(self, dt = 0.1, 
                       alpha1 = 0.1, 
                       alpha2 = 0.01, 
                       alpha3 = 0.01, 
                       alpha4 = 0.1,
                       sig_r = 0.1,
                       sig_ph = 0.05,
                       landmarks = np.array([[6,4],[-7,8],[6,-4]])):
        self.dt = dt;
        self.alpha1 = alpha1;
        self.alpha2 = alpha2;
        self.alpha3 = alpha3;
        self.alpha4 = alpha4;
        self.sig_r = sig_r;
        self.sig_ph = sig_ph;
        self.landmarks = landmarks;


    def EKF_Localization(self, mu, Sig, u, state): #need to make acccept z instead of state
        #mu in the last time step 
        lmdModel = lmd(self.sig_r, self.sig_ph);
        lmdMeasured = np.zeros((np.size(self.landmarks,0),2));
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
        V = np.zeros((3,2));
        V[0][0] = ( -np.sin(theta) + np.sin(theta + wc*self.dt) ) / wc;
        V[0][1] = ( vc * (np.sin(theta) - np.sin(theta + wc*self.dt)) ) / wc**2 + ( vc*np.cos(theta + wc*self.dt)*self.dt ) / wc; 
        V[1][0] = ( np.cos(theta) - np.cos(theta + wc*self.dt) ) / wc;
        V[1][1] = - ( vc * (np.cos(theta) - np.cos(theta+wc*self.dt)) ) / wc**2 + ( vc * np.sin(theta + wc*self.dt)*self.dt ) / wc;
        V[2][1] = self.dt;
        #control noise covariance
        M = np.zeros((2,2));
        M[0][0] = self.alpha1*vc**2 + self.alpha2*wc**2;
        M[1][1] = self.alpha3*vc**2 + self.alpha4*wc**2;
        #state estimate - prediction step
        mu_x = mu_x - vc*np.sin(theta)/wc + vc*np.sin(theta+wc*self.dt)/wc;
        mu_y = mu_y + vc*np.cos(theta)/wc - vc*np.cos(theta+wc*self.dt)/wc;
        mu_th = mu_th + wc*self.dt;
        mu_est = np.array([mu_x,mu_y,mu_th]);
        #state covariance - prediction step
        Sig_est = np.dot( G ,np.dot(Sig,np.transpose(G)) ) + np.dot( V, np.dot(M,np.transpose(V)) );
        #Uncertainty due to measurement noise
        Q = np.zeros((2,2));
        Q[0][0] = self.sig_r**2;
        Q[1][1] = self.sig_ph**2;
        #Measurement Update
        num_landmarks = np.size(self.landmarks,0);
        for i in range(0,num_landmarks):
            landmark = self.landmarks[i]
            Range = lmdModel.getRange(landmark,state);
            Bearing = lmdModel.getBearing(landmark,state);
            z = np.array([[Range],[Bearing]]);
            lmdMeasured[i] = lmdModel.getGlobalXY(Range,Bearing,state);
            q = (landmark[0] - mu[0])**2 + (landmark[1] - mu[1])**2;
            b = np.arctan2(landmark[1] - mu[1], landmark[0] - mu[0]) - mu[2];
            z_hat = np.array([[np.sqrt(q)] , [b]]);
            H = np.zeros((2,3));
            H[0][0] = -(landmark[0] - mu[0])/np.sqrt(q);
            H[0][1] = -(landmark[1] - mu[1])/np.sqrt(q);
            H[1][0] = (landmark[1] - mu[1])/q;
            H[1][1] = -(landmark[0] - mu[0])/q;
            H[1][2] = -1.0;
            S = np.dot( H , np.dot(Sig_est,np.transpose(H)) ) + Q;
            K = np.dot( Sig_est , np.dot(np.transpose(H) , np.linalg.inv(S)) );
            mu_est = mu_est.reshape(-1,1) + np.dot(K,(z-z_hat));
            mu_est = mu_est.flatten();
            Sig_est = np.dot( (np.identity(3) - np.dot(K,H)) , Sig_est);
        return mu_est, Sig_est, lmdMeasured

        
