#givenParameters 
from scipy.io import loadmat
import numpy as np

matdata = loadmat('hw3_1_soln_data.mat')
#matdata = loadmat('hw3_1_soln_data.mat')
#matdata = loadmat('hw3_1_soln_data.mat')
vc = matdata['v'][0]
wc = matdata['om'][0]
t = matdata['t'][0]
x_given = matdata['x'][0]
y_given = matdata['y'][0]
theta_given = matdata['th'][0]
#parameters file
dt = t[1] - t[0]
T = t[np.size(t) - 1]
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1
x0 = x_given[0]
y0 = y_given[0]
theta0 = theta_given[0]
Sig0 = np.array([[.1,0,0],
               [0,.1,0],
               [0,0,0.1]])
alfa = 0.4
kappa = 4.0
beta = 2
sig_r = 0.1
sig_b = 0.05
m = np.array([[6,4],[-7,8],[6,-4]]) #landmark locations
animation_speed = 100;
cycle = True;
given = True;



