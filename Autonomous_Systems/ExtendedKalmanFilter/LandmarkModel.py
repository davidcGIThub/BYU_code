#landmark model
import numpy as np 

class LandmarkModel:

    def __init__(self, std_r = 0.1, std_b = 0.05):
        self.std_r = std_r;
        self.std_b = std_b;

    def getXYdist(self, landmark, reference):
        return landmark - reference[0:2];

    #reference should be a 1X3 numpy array
    def getRange(self, landmark, reference):
        XY_dist = self.getXYdist(landmark,reference);
        Range = np.sqrt(np.sum(XY_dist**2)) + np.random.randn()*self.std_r;
        return Range

    def getBearing(self, landmark, reference):
        XY_dist = self.getXYdist(landmark,reference);
        theta = reference[2];
        Bearing = np.arctan2(XY_dist[1],XY_dist[0]) - theta + np.random.randn()*self.std_b;
        return Bearing

    def getGlobalXY(self, Range, Bearing, reference):
        x = Range*np.cos(Bearing + reference[2]) + reference[0];
        y = Range*np.sin(Bearing + reference[2]) + reference[1];
        return np.array([x,y])
