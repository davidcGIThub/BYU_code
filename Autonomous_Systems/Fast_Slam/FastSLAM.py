#ExtendedKalmanFilter 
import numpy as np 

class Fast_SLAM:

    def __init__(self, dt, alpha, sig_r, sig_ph, pose_noise):
        self.dt = dt
        self.alpha1 = alpha[0]
        self.alpha2 = alpha[1]
        self.alpha3 = alpha[2]
        self.alpha4 = alpha[3]
        self.sig_r = sig_r
        self.sig_ph = sig_ph
        self.pose_noise = pose_noise #propogation noise particle pose

    def low_variance_sampler(self, ki_bar, w, M):
        r = np.random.uniform(0, 1.0 / float(M))
        c = w[0]
        i = 0
        ki = ki_bar * 0
        for k in range(1, M + 1):
            U = r + (k - 1) / float(M)
            while U > c:
                i = i + 1
                c = c + w[i]
            ki[k - 1,:] = ki_bar[i, :]
        return ki

    def fast_SLAM_1(self, z, c, u, Y, detected):
        M = np.size(Y,0)
        N = np.size(z,0)
        Y_new = np.copy(Y)
        vc = u[0]
        wc = u[1]
        w = np.ones(M)
        #loop through all the particles
        for i in range(0,M):
            #estimate new pose for particle
            v_hat = vc + (self.alpha1 * vc**2 + self.alpha2 * wc**2) * np.random.randn()
            w_hat = wc + (self.alpha3 * vc**2 + self.alpha4 * wc**2) * np.random.randn()
            x = Y[i][0]
            y = Y[i][1]
            theta = Y[i][2]
            x = x - v_hat*np.sin(theta)/w_hat + v_hat*np.sin(theta+w_hat*self.dt)/w_hat 
            y = y + v_hat*np.cos(theta)/w_hat - v_hat*np.cos(theta+w_hat*self.dt)/w_hat 
            theta = theta + w_hat*self.dt 
            #loop through all the features
            for j in range(0,N):
                #if feature was detected
                if c[j] == True:
                    #measurement calculations
                    Range = z[j,0]
                    Bearing = z[j,1]
                    Qt = np.zeros((2,2))
                    Qt[0][0] = self.sig_r**2
                    Qt[1][1] = self.sig_ph**2
                    #if the feature was never detected
                    if detected[j] == False:
                        #initialize the mean for feature
                        mu_x = x + Range*np.cos(Bearing+theta)
                        mu_y = y + Range*np.sin(Bearing+theta)
                        #Calculate Jacobian
                        q = (mu_x - x)**2 + (mu_y - y)**2
                        H = np.zeros((2,2))
                        H[0][0] = (mu_x - x)/np.sqrt(q)
                        H[0][1] = (mu_y - y)/np.sqrt(q)
                        H[1][0] = -(mu_y - y)/q
                        H[1][1] = (mu_x - x)/q
                        Hinv = np.linalg.inv(H)
                        #initialize covariance for feature
                        Sigma = np.dot( Hinv , np.dot(Qt,np.transpose(Hinv)) )

                    else:
                        mu_x = Y[i][3+6*j]
                        mu_y = Y[i][4+6*j]
                        Sigma = np.array([ [ Y[i][5+6*j] , Y[i][6+6*j] ],
                                           [ Y[i][7+6*j] , Y[i][8+6*j] ]])
                        #measurement prediction
                        q = (mu_x - x)**2 + (mu_y - y)**2
                        b = np.arctan2(mu_y - y, mu_x - x) - theta
                        z_hat = np.array([[np.sqrt(q)] , [b]])
                        z_diff = np.array([[Range] , [Bearing]]) - z_hat
                        z_diff[1,0] -= np.pi * 2 * np.floor((z_diff[1,0] + np.pi) / (2 * np.pi))
                        #calculate Jacobian
                        H = np.zeros((2,2))
                        H[0][0] = (mu_x - x)/np.sqrt(q)
                        H[0][1] = (mu_y - y)/np.sqrt(q)
                        H[1][0] = -(mu_y - y)/q
                        H[1][1] = (mu_x - x)/q
                        #Measurement covariance
                        Q = np.dot(H , np.dot(Sigma , np.transpose(H))) + Qt
                        #calculate Kalman Gain
                        K = np.dot(Sigma, np.dot(np.transpose(H) , np.linalg.inv(Q)))
                        #update feature mean
                        mu = np.array([[mu_x],[mu_y]]) + np.dot(K,z_diff)
                        mu_x = mu[0,0]
                        mu_y = mu[1,0]
                        #update feature covariance
                        Sigma = np.dot( (np.eye(2) - np.dot(K,H)) , Sigma)
                        #update importance factor
                        w[i] = w[i] * np.linalg.det(2*np.pi*Q)**(-1/2) * np.exp( -1/2 * np.dot(z_diff.flatten() , np.dot(np.linalg.inv(Q),z_diff)) )
                else:
                    #leave unchanged
                    mu_x = Y[i][3+6*j]
                    mu_y = Y[i][4+6*j]
                    Sigma = np.array([ [ Y[i][5+6*j] , Y[i][6+6*j] ],
                                       [ Y[i][7+6*j] , Y[i][8+6*j] ]])
                #assign feature values to new particles matrix
                Y_new[i][3+6*j] = mu_x
                Y_new[i][4+6*j] = mu_y
                Y_new[i][5+6*j] = Sigma[0,0]
                Y_new[i][6+6*j] = Sigma[0,1]
                Y_new[i][7+6*j] = Sigma[1,0]
                Y_new[i][8+6*j] = Sigma[1,1]
        #what if no landmarks are seen?????
            Y_new[i][0] = x
            Y_new[i][1] = y
            Y_new[i][2] = theta
        # Resampling
        w = w / np.sum(w)  # normalize the weights
        Y_new = self.low_variance_sampler(Y_new, w, M)  # particles
        return Y_new