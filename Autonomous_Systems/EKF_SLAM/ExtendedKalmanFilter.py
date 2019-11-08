#ExtendedKalmanFilter
import numpy as np 

class EKF:

    def __init__(self, dt, alpha, sig_r, sig_ph)
        self.dt = dt
        self.alpha = alpha
        self.sig_r = sig_r
        self.sig_ph = sig_ph

    def EKF_Localization(self, mu, Sig, u, z, landmarks)
        #prior estimated state
        x = mu[0]
        y = mu[1]
        theta = mu[2]
        #control input
        vc = u[0]
        wc = u[1]
        #constants
        alpha1 = alpha[0]
        alpha2 = alpha[1]
        alpha3 = alpha[2]
        alpha4 = alpha[3]
        #jacobian of g(u(t),x(t-1))
        G = np.array([ [0   ,   vc/wc*np.cos(theta) + vc/wc*np.cos(theta+wc*self.dt)],
                       [0   ,  -vc/wc*np.sin(theta) + vc/wc*np.sin(theta+wc*self.dt)]])
        #Jacobian to map noise from control space to state space
        V = np.array([[(-np.sin(theta) + np.sin(theta + wc*self.dt) ) / wc  ,    ( vc * (np.sin(theta) - np.sin(theta + wc*self.dt)) ) / wc**2 + ( vc*np.cos(theta + wc*self.dt)*self.dt ) / wc],
                      [( np.cos(theta) - np.cos(theta + wc*self.dt) ) / wc  ,   -( vc * (np.cos(theta) - np.cos(theta+wc*self.dt)) ) / wc**2 + ( vc * np.sin(theta + wc*self.dt)*self.dt ) / wc],
                      [self.dt                                              ,   0]])
        #control noise covariance
        M = np.array([[self.alpha1*vc**2 + self.alpha2*wc**2,   0],
                      [0    ,   self.alpha3*vc**2 + self.alpha4*wc**2]])
        #state estimate - prediction step
        mu_est = np.zeros(3)
        mu_est[0] = x - vc*np.sin(theta)/wc + vc*np.sin(theta+wc*self.dt)/wc
        mu_est[1] = y + vc*np.cos(theta)/wc - vc*np.cos(theta+wc*self.dt)/wc
        mu_est[2] = theta + wc*self.dt
        #state covariance - prediction step
        Sig_est = np.dot( G ,np.dot(Sig,np.transpose(G)) ) + np.dot( V, np.dot(M,np.transpose(V)) )
        #Uncertianty due to measurement noise
        Q = np.array([[self.sig_r**2    ,   0],
                      [0    ,   self.sig_ph**2]])
        #Measurement Update
        num_landmarks = np.size(self.landmarks,0)
        for i in range(0,num_landmarks):
            landmark = landmarks[i]
            Range = z[i][0]
            Bearing = z[i][1]
            q = (landmark[0] - mu[0])**2 + (landmark[1] - mu[1])**2;
            b = np.arctan2(landmark[1] - mu[1], landmark[0] - mu[0]) - mu[2];
            z_hat = np.array([[np.sqrt(q)] , [b]]);
            H = np.array([[-(landmark[0] - mu[0])/np.sqrt(q) ,   -(landmark[1] - mu[1])/np.sqrt(q)   ,   0],
                          [-(landmark[1] - mu[1])/q          ,   -(landmark[0] - mu[0])/q            ,   -1.0]])
            S = np.dot( H , np.dot(Sig_est,np.transpose(H)) ) + Q
            K = np.dot( Sig_est , np.dot(np.transpose(H) , np.linalg.inv(S)) )
            mu_est = ( mu_est[:,None] + np.dot(K,(z-z_hat)) ).flatten()
            Sig_est = np.dot( (np.identity(3) - np.dot(K,H)) , Sig_est)
    return mu_est, Sig_est