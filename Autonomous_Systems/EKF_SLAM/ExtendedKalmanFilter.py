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

    def EKF_SLAM(self, mu, sigma, u, z, c, detected_flag):
        N = np.size(z,0)
        vc = u[0]
        wc = u[1]
        F = np.concatenate( (np.identity(3) , np.zeros((3,2*N))) , 1 )
        motion = np.array([[-(vc/wc)*np.sin(mu[2]) + (vc/wc)*np.sin(mu[2]+wc*self.dt)],
                           [(vc/wc)*np.cos(mu[2]) - (vc/wc) *np.cos(mu[2]+wc*self.dt)],
                           [wc*self.dt]])
        mu_bar = (mu[:,None] + np.dot( np.transpose(F) , motion)).flatten()
        cov_motion = np.array([[ 0 , 0 , -(vc/wc)*np.cos(mu[2]) + (vc/wc)*np.cos(mu[2] + wc*self.dt) ],
                               [ 0 , 0 , -(vc/wc)*np.sin(mu[2]) + (vc/wc)*np.sin(mu[2] + wc*self.dt) ],
                               [ 0 , 0 , 0]])
        G = np.identity(3+2*N) + np.dot( np.transpose(F) , np.dot(cov_motion,F))
        #Jacobian to map noise from control space to state space
        V = np.zeros((3,2))
        V[0][0] = ( -np.sin(mu[2]) + np.sin(mu[2] + wc*self.dt) ) / wc
        V[0][1] = ( vc * (np.sin(mu[2]) - np.sin(mu[2] + wc*self.dt)) ) / wc**2 + ( vc*np.cos(mu[2] + wc*self.dt)*self.dt ) / wc
        V[1][0] = ( np.cos(mu[2]) - np.cos(mu[2] + wc*self.dt) ) / wc
        V[1][1] = - ( vc * (np.cos(mu[2]) - np.cos(mu[2]+wc*self.dt)) ) / wc**2 + ( vc * np.sin(mu[2] + wc*self.dt)*self.dt ) / wc
        V[2][1] = self.dt
        #control noise covariance
        M = np.zeros((2,2))
        M[0][0] = self.alpha1*vc**2 + self.alpha2*wc**2
        M[1][1] = self.alpha3*vc**2 + self.alpha4*wc**2
        R = np.dot(V , np.dot(M , np.transpose(V)))
        sigma_bar = np.dot(G , np.dot(sigma , np.transpose(G)) + np.dot(np.transpose(F) , np.dot(R,F)) )
        #Uncertainty due to measurement noise
        Q = np.zeros((2,2))
        Q[0][0] = self.sig_r**2
        Q[1][1] = self.sig_ph**2
        for i in range(0,N):
            if c[i] == True:
                Range = z[i,0]
                Bearing = z[i,1]
                Bearing -= np.pi * 2 * np.floor((Bearing + np.pi) / (2 * np.pi))
                mu_x_land = mu_bar[3+2*i]
                mu_y_land = mu_bar[4+2*i]
                if detected_flag[i] == False:
                    mu_x_land = mu_bar[3+2*i] = mu_bar[0] + Range*np.cos(Bearing+mu_bar[2])
                    mu_y_land = mu_bar[4+2*i] = mu_bar[1] + Range*np.sin(Bearing+mu_bar[2])
                delta = np.array([mu_x_land - mu_bar[0],
                                  mu_y_land - mu_bar[1]])
                q = np.dot(delta , delta[:,None])[0]
                z_hat = np.array([[np.sqrt(q)],
                                  [np.arctan2(delta[1],delta[0]) - mu_bar[2]]])
                Fi = np.zeros((5,3+2*N))
                Fi[0,0] = Fi[1,1] = Fi[2,2] = 1.0
                Fi[3,3+2*i] = Fi[4,4+2*i] = 1.0
                H_intermediate = np.array([[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0 , np.sqrt(q)*delta[0] , np.sqrt(q)*delta[1]],
                                           [delta[1]            , -delta[0]           ,-q , -delta[1]           , delta[0]]])
                H = (1/q) * np.dot(H_intermediate , Fi)
                K_intermediate = np.dot(H , np.dot(sigma_bar , np.transpose(H))) + Q
                K_intermediate = np.linalg.inv(K_intermediate)
                K = np.dot(sigma_bar , np.dot(np.transpose(H) , K_intermediate))
                z_diff = z[i][:,None] - z_hat
                z_diff[1,0] -= np.pi * 2 * np.floor((z_diff[1,0] + np.pi) / (2 * np.pi))
                mu_bar = (mu_bar[:,None] + np.dot(K, z_diff)).flatten()
                rows = np.size(K,0)
                sigma_bar = np.dot( np.identity(rows)-np.dot(K,H) , sigma_bar)
        return mu_bar, sigma_bar

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
            H[0][0] = -(landmark[0] - mu[0])/np.sqrt(q)
            H[0][1] = -(landmark[1] - mu[1])/np.sqrt(q)
            H[1][0] = (landmark[1] - mu[1])/q
            H[1][1] = -(landmark[0] - mu[0])/q
            H[1][2] = -1.0
            S = np.dot( H , np.dot(Sig_est,np.transpose(H)) ) + Q
            K = np.dot( Sig_est , np.dot(np.transpose(H) , np.linalg.inv(S)) )
            N = np.size(z,0)
            mu_est = mu_est.reshape(-1,1) + np.dot(K,(z[i][:,None]-z_hat))
            mu_est = mu_est.flatten()
            Sig_est = np.dot( (np.identity(3) - np.dot(K,H)) , Sig_est)
        return mu_est, Sig_est