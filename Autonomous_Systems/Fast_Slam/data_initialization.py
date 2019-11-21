#data initialization file
import numpy as np

#time parameters
sec = 20
dt = 0.1
t = np.linspace(0,sec,sec/dt+1)

#Grid layout
x_limits = 20
y_limits = 20
ms = 5 #landmark size

#initial conditions
x0 = -5.0 #m
y0 = -3.0 #m
theta0 = np.pi/2.0 #rad
state = np.array([x0,y0,theta0])
pose = np.array([x0,y0,theta0])
N = 5 #num landmarks
landmarks = np.random.uniform(-x_limits+1,x_limits-1,(N,2))
M = 100 #number of particles

#tuning parameters
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1
alpha = np.array([alpha1,alpha2,alpha3,alpha4])
sig_r = 0.1
sig_b = 0.05
pose_noise = np.array([0.5,0.1])

#Measurement Parameters
fov = 360
fov = np.pi*fov/180.0

#initialize the particles
pose_init = np.tile(pose,(M,1))
feature_init = np.tile(np.array([x_limits-.5 , y_limits-.5, np.inf, 0, 0, np.inf]),(M,N)) * np.random.uniform(-1,1,(M,N*6))
feature_init[feature_init == np.inf] = np.exp(50)
feature_init[feature_init == -np.inf] = np.exp(50)
Y = np.concatenate((pose_init, feature_init),1)
features = np.copy(landmarks)
for i in range(0,N):
        features[i,0] = np.sum(Y[:,3+i*6])/M
        features[i,1] = np.sum(Y[:,4+i*6])/M

#Detection flags
c = np.ones(N)
detected_flag = np.zeros(N)

#Input commands
vc = 1 + 0.5*np.cos(2.0*np.pi*.2*t)
wc = -0.2 + 2*np.cos(2*np.pi*0.6*t)

#storage vectors
x_true = t * 0
y_true = t * 0
theta_true = t * 0
x_est = t * 0
y_est = t * 0
theta_est = t * 0