#landmark model
import numpy as np 

class LandmarkModel:

    def __init__(self,
                 std_r = 0.1,
                 std_b = 0.05):
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

    def getBearings(self, reference,m):
        len_m = np.size(m,0)
        XY_dist = self.getXYdist(reference,m)
        theta = reference[2]
        bearings = np.arctan2(XY_dist[:,1],XY_dist[:,0])
        bearings = bearings.reshape(-1,1) - theta
        bearings = bearings + np.random.randn(len_m,1)*self.std_b
        return bearings

    def getLandmarks(self,reference,m):
        bearings = self.getBearings(reference,m)
        ranges = self.getRanges(reference,m)
        x = ranges*np.cos(bearings + reference[2]) + reference[0]
        y = ranges*np.sin(bearings + reference[2]) + reference[1]
        return np.concatenate((x,y),1)
