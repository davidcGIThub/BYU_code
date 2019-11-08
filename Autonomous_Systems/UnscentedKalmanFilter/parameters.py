import numpy as np
#parameters file
dt = 0.1
T = 20.0
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1
x0 = -5.0 #m
y0 = -3.0 #m
theta0 = np.pi/2.0 #rad
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
given = False;
#calculated parameters
t = np.linspace(0,T,T/dt+1)
vc = 1 + 0.5*np.cos(2.0*np.pi*.2*t)
wc = -0.2 + 2*np.cos(2*np.pi*0.6*t)