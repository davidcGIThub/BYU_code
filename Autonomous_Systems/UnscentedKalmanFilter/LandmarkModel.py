#landmark model
import numpy as np 

class LandmarkModel:

    def __init__(self,
                 locations = np.array([[6,4],[-7,8],[6,-4]]),
                 std_r = 0.1,
                 std_b = 0.05):
        self.locations = locations;
        self.std_r = std_r;
        self.std_b = std_b;
        self.len = np.size(self.locations,0);

    def getTrueLandmarks(self):
        return self.locations;

    def getXYdist(self,reference):
        return self.locations - reference[0:2];

    #reference should be a 1X3 numpy array
    def getRanges(self, reference, meas_noise = False):
        XY_dist = self.getXYdist(reference);
        axis = np.size(np.shape(XY_dist)) - 1;
        ranges = np.sqrt(np.sum(XY_dist**2,axis));
        ranges = ranges.reshape(-1,1);
        if meas_noise:
            ranges = ranges + np.random.randn(self.len,1)*self.std_r;
        return ranges

    def getBearings(self, reference, meas_noise = False):
        XY_dist = self.getXYdist(reference);
        theta = reference[2];
        bearings = np.arctan2(XY_dist[:,1],XY_dist[:,0])
        bearings = bearings.reshape(-1,1) - theta;
        if meas_noise:
            bearings = bearings + np.random.randn(self.len,1)*self.std_b;
        return bearings

    def getLandmarks(self,reference):
        bearings = self.getBearings(reference,True);
        ranges = self.getRanges(reference,True);
        x = ranges*np.cos(bearings + reference[2]) + reference[0];
        y = ranges*np.sin(bearings + reference[2]) + reference[1];
        return np.concatenate((x,y),1)
