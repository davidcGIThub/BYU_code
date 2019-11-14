#ExtendedKalmanFilter 
import numpy as np 

class EKF:

    def __init__(self, dt, alpha, sig_r, sig_ph):
        self.dt = dt
        self.alpha1 = alpha[0]
        self.alpha2 = alpha[1]
        self.alpha3 = alpha[2]
        self.alpha4 = alpha[3]
        self.sig_r = sig_r
        self.sig_ph = sig_ph

    def EKF_Localization(self, mu, Sig, u, z, landmarks): #need to make acccept z instead of state
        #mu in the last time step 
        mu_x = mu[0]
        mu_y = mu[1]
        mu_th = mu[2]
        #control input
        vc = u[0]
        wc = u[1]
        #use prior theta to predict current state
        theta = mu_th
        #jacobian of g(u(t),x(t-1))
        G = np.identity(3)
        G[0][2] = -vc/wc*np.cos(theta) + vc/wc*np.cos(theta+wc*self.dt)
        G[1][2] = -vc/wc*np.sin(theta) + vc/wc*np.sin(theta+wc*self.dt)
        #Jacobian to map noise from control space to state space
        V = np.zeros((3,2))
        V[0][0] = ( -np.sin(theta) + np.sin(theta + wc*self.dt) ) / wc
        V[0][1] = ( vc * (np.sin(theta) - np.sin(theta + wc*self.dt)) ) / wc**2 + ( vc*np.cos(theta + wc*self.dt)*self.dt ) / wc
        V[1][0] = ( np.cos(theta) - np.cos(theta + wc*self.dt) ) / wc
        V[1][1] = - ( vc * (np.cos(theta) - np.cos(theta+wc*self.dt)) ) / wc**2 + ( vc * np.sin(theta + wc*self.dt)*self.dt ) / wc
        V[2][1] = self.dt
        #control noise covariance
        M = np.zeros((2,2))
        M[0][0] = self.alpha1*vc**2 + self.alpha2*wc**2
        M[1][1] = self.alpha3*vc**2 + self.alpha4*wc**2
        #state estimate - prediction step
        mu_x = mu_x - vc*np.sin(theta)/wc + vc*np.sin(theta+wc*self.dt)/wc
        mu_y = mu_y + vc*np.cos(theta)/wc - vc*np.cos(theta+wc*self.dt)/wc
        mu_th = mu_th + wc*self.dt
        mu_est = np.array([mu_x,mu_y,mu_th])
        #state covariance - prediction step
        Sig_est = np.dot( G ,np.dot(Sig,np.transpose(G)) ) + np.dot( V, np.dot(M,np.transpose(V)) )
        #Uncertainty due to measurement noise
        Q = np.zeros((2,2))
        Q[0][0] = self.sig_r**2
        Q[1][1] = self.sig_ph**2
        #Measurement Update
        num_landmarks = np.size(landmarks,0)
        for i in range(0,num_landmarks):
            landmark = landmarks[i]
            q = (landmark[0] - mu[0])**2 + (landmark[1] - mu[1])**2
            b = np.arctan2(landmark[1] - mu[1], landmark[0] - mu[0]) - mu[2]
            z_hat = np.array([[np.sqrt(q)] , [b]])
            H = np.zeros((2,3))
            H[0][0] = -(landmark[0] - mu[0])/np.sqrt(q)N = np.size(z,0))
            mu_est = mu_est.reshape(-1,1) + np.dot(K,(z[i][:,None]-z_hat))
            mu_est = mu_est.flatten()
            Sig_est = np.dot( (np.identity(3) - np.dot(K,H)) , Sig_est)
        return mu_est, Sig_est

    def EKF_SLAM(mu, sigma, u, z, c)
        N = np.size(z,0)
        vc = u[0]
        wc = u[1]
        mu_x = mu[0]
        mu_y = mu[1]
        mu_theta = mu[2]
        F = np.concatenate(np.identity(3),np.zeros(3,2*N))
        motion = np.array([[-(vc/wc)*np.sin(mu_theta) + (vc/wc)*np.sin(mu_theta+wc*self.dt)],
                           [(vc/wc)*np.cos(mu_theta) - (vc/wc) *np.cos(mu_theta+wc*self.dt)],
                           [wc*self.dt]])
        mu_bar = mu[:,None] + np.dot( np.transpose(F) , motion)
        cov_motion = np.array([[ 0 , 0 , -(vc/wc)*np.cos(mu_theta) + (vc/wc)*np.cos(mu_theta + wc*dt) ],
                               [ 0 , 0 , -(vc/wc)*np.sin(mu_theta) + (vc/wc)*np.sin(mu_theta + wc*dt) ],
                               [ 0 , 0 , 0]])
        G = np.identity(3+2*N) + np.dot( np.transpose(F) , np.dot(cov_motion,F))
        #Jacobian to map noise from control space to state space
        V = np.zeros((3,2))
        V[0][0] = ( -np.sin(mu_theta) + np.sin(mu_theta + wc*self.dt) ) / wc
        V[0][1] = ( vc * (np.sin(mu_theta) - np.sin(mu_theta + wc*self.dt)) ) / wc**2 + ( vc*np.cos(mu_theta + wc*self.dt)*self.dt ) / wc
        V[1][0] = ( np.cos(mu_theta) - np.cos(mu_theta + wc*self.dt) ) / wc
        V[1][1] = - ( vc * (np.cos(mu_theta) - np.cos(mu_theta+wc*self.dt)) ) / wc**2 + ( vc * np.sin(mu_theta + wc*self.dt)*self.dt ) / wc
        V[2][1] = self.dt
        #control noise covariance
        M = np.zeros((2,2))
        M[0][0] = self.alpha1*vc**2 + self.alpha2*wc**2
        M[1][1] = self.alpha3*vc**2 + self.alpha4*wc**2
        R = np.dot(V , np.dot(M , np.transpose(V)))
        sigma_bar = np.dot(G , np.dot(sigma , np.transpose(G)) + np.dot(np.transpose(F) , np.dot(R,F))
        #Uncertainty due to measurement noise
        Q = np.zeros((2,2))
        Q[0][0] = self.sig_r**2
        Q[1][1] = self.sig_ph**2
        for i in range(0,N):
            if c[i] == True:

            else



