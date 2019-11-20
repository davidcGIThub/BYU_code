#data initialization file
import numpy as np
sec = 60
t = np.linspace(0,sec,sec/0.1+1)
x_true = t * 0
y_true = t * 0
theta_true = t * 0
x_est = t * 0
y_est = t * 0
theta_est = t * 0

dt = 0.1
vc = 1 + 0.5*np.cos(2.0*np.pi*.2*t)
wc = -0.2 + 2*np.cos(2*np.pi*0.6*t)
x0 = -5.0 #m
y0 = -3.0 #m
theta0 = np.pi/2.0 #rad
state = np.array([x0,y0,theta0])
mu = np.array([x0,y0,theta0])
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1
alpha = np.array([alpha1,alpha2,alpha3,alpha4])
step = 0
sig_r = 0.1
sig_b = 0.05

x_limits = 20
y_limits = 20
ms = 5 #landmark size

N = 14 #number of landmarks
landmarks = np.random.uniform(-x_limits+1,x_limits-1,(N,2))
mu = np.zeros(3+2*N)
mu[0] = x0
mu[1] = y0
mu[2] = theta0
Sig = np.exp(100.0)*np.identity(2*N + 3)
Sig[0,0] = 0.0
Sig[1,1] = 0.0
Sig[2,2] = 0.0
cov = np.zeros((3+2*N,np.size(t)))

c = np.ones(N)
detected_flag = np.zeros(N)
fov = 360

fov = np.pi*fov/180.0