#LandMark Model
import numpy as np 

class MeasurementModel:

    def __init__(self,
                 std_r = 0.2,
                 std_b = 0.1):
        self.std_r = std_r
        self.std_b = std_b

    def getXYdist(self,reference,m):
        return m - reference[0:2]

    #reference should be a 1X3 numpy array
    def getRanges(self, reference,m):
        len_m = np.size(m,0)
        XY_dist = self.getXYdist(reference,m)
        axis = np.size(np.shape(XY_dist)) - 1
        ranges = np.sqrt(np.sum(XY_dist**2,axis))
        ranges = ranges.reshape(-1,1)
        ranges = ranges + np.random.randn(len_m,1)*self.std_r
        return ranges

    def getBearings(self, reference,m,fov):
        len_m = np.size(m,0)
        XY_dist = self.getXYdist(reference,m)
        theta = reference[2]
        bearings = np.arctan2(XY_dist[:,1],XY_dist[:,0])
        bearings = bearings.reshape(-1,1) - theta
        bearings = bearings + np.random.randn(len_m,1)*self.std_b
        c = bearings.flatten()
        c -= np.pi * 2 * np.floor((c + np.pi) / (2 * np.pi))
        c[np.abs(c) > fov/2.0] = 0
        c[np.abs(c) > 0] = 1
        return bearings, c.astype(int)

    def getLandmarkEstimates(self,reference,ranges,bearings):
        x = ranges*np.cos(bearings + reference[2]) + reference[0]
        y = ranges*np.sin(bearings + reference[2]) + reference[1]
        return np.concatenate((x,y),1)

    def getCovariancePoints(self,m_est,cov):
        N = int(np.size(m_est)/2)
        num_pts = 20
        points = np.zeros((4*N,2))
        points2 = np.zeros((num_pts*N,2))
        for i in range(0,N):
            P = np.array([[cov[2*i], 0],
                          [0 , cov[2*i+1]]])
            (U,S,V) = np.linalg.svd(P)
            C = np.dot(U, np.sqrt(S)).flatten()
            theta = np.linspace(0,2*np.pi,20)
            circle = np.transpose(np.array([np.cos(theta), np.sin(theta)]))
            ellipse = C*circle + np.array([m_est[2*i] , m_est[2*i+1]])
            points2[num_pts*i:num_pts*i+num_pts] = ellipse
            #x_values
            points[4*i+0,0] = m_est[2*i] + cov[2*i] 
            points[4*i+1,0] = m_est[2*i] - cov[2*i]
            points[4*i+2,0] = m_est[2*i]
            points[4*i+3,0] = m_est[2*i]
            #y_values
            points[4*i+0,1] = m_est[2*i+1]
            points[4*i+1,1] = m_est[2*i+1] 
            points[4*i+2,1] = m_est[2*i+1] + cov[2*i+1]
            points[4*i+3,1] = m_est[2*i+1] - cov[2*i+1]
        return points2
