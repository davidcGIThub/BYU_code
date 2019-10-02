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

    def getLocations(self):
        return self.locations;

    #reference should be a 1X3 numpy array

    def estimateRanges(self, reference):
        XY_dist = self.locations - reference[0:2];
        axis = np.size(np.shape(XY_dist)) - 1;
        ranges = np.sqrt(np.sum(XY_dist**2,axis));
        ranges = ranges.reshape(-1,1);
        est_ranges = ranges + np.random.randn(self.len,1)*self.std_r;
        return est_ranges

    def estimateBearings(self, reference):
        XY_dist = self.locations - reference[0:2];
        bearings = np.arctan2(XY_dist[:,1],XY_dist[:,0])
        bearings = bearings.reshape(-1,1);
        theta = reference[2];
        est_bearings = bearings - theta + np.random.randn(self.len,1)*self.std_b;
        return est_bearings

    def estimateLocations(self,reference):
        est_bearings = self.estimateBearings(reference);
        est_ranges = self.estimateRanges(reference);
        x_est = est_ranges*np.cos(est_bearings + reference[2]) + reference[0];
        y_est = est_ranges*np.sin(est_bearings + reference[2]) + reference[1];
        return np.concatenate((x_est,y_est),1)
