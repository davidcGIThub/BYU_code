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

    #reference should be a 1X2 numpy array
    def getRanges(self,reference):
        XY_dist = self.locations - reference;
        axis = np.size(np.shape(XY_dist)) - 1;
        ranges = np.sqrt(np.sum(XY_dist**2,axis));
        ranges = ranges.reshape(-1,1);
        return ranges

    def estimateRanges(self, reference):
        ranges =  self.getRanges(reference);
        est_ranges = ranges + np.random.randn(self.len,1)*self.std_r;
        return est_ranges

    def getBearings(self,reference):
        bearings = np.zeros([self.len,1]);
        self.locations;
        XY_dist = self.locations - reference;
        for i in range(0,self.len):
            bearings[i] = np.arctan2(XY_dist[i][1],XY_dist[i][0]);
        return bearings

    def estimateBearings(self, reference):
        bearings =  self.getBearings(reference);
        est_bearings = bearings + np.random.randn(self.len,1)*self.std_b;
        return est_bearings

    def estimateLocations(self,reference):
        est_bearings = self.estimateBearings(reference);
        est_ranges = self.estimateRanges(reference);
        x_est = est_ranges*np.cos(est_bearings) + reference[0];
        y_est = est_ranges*np.sin(est_bearings) + reference[1];
        return np.concatenate((x_est,y_est),1)
