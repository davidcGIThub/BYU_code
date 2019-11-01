#data initialization
import numpy as np
from scipy.io import loadmat

#load matlab data
mat_data = loadmat('midterm_data.mat')

#estimate data
t = mat_data['t'].flatten()
landmarks = np.transpose(mat_data['m'])
w = mat_data['om'].flatten()
v = mat_data['v'].flatten()

x0 = -5 #m
y0 = 0 #m
theta0 = np.pi/2.0 #rad

sig_r = 0.2 #m
sig_phi = 0.1 #rad
sig_v = 0.15 #m/s
sig_w = 0.1 #rad/s
dt = t[1]-t[0] #

#truth data
v_tr = mat_data['v_c'].flatten()
w_tr = mat_data['om_c'].flatten()
X_tr = mat_data['X_tr']
x_tr = X_tr[0,:].flatten()
y_tr = X_tr[1,:].flatten()
theta_tr = X_tr[2,:].flatten()
bearing_tr = mat_data['bearing_tr']
range_tr = mat_data['range_tr']

