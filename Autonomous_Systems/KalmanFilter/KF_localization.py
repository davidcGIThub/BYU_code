import numpy as np 
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt

matdata = loadmat('hw1_soln_data.mat');
z = matdata['z'];
Ts = 0.05; #sec
t = np.linspace(0,50,50/0.05+1); #sec
F = t*0; #N
F[t<5] = 50;
F[t>25] = -50;
F[t>30] = 0;
m = 100; #kg
b = 20; #N-s/m
Ts = 0.05; #s

#state Space Equation
#x_dot = Ax + Bu
#z = Cx + Du

Ac = np.array([[0 , 1],
              [0 , -b/m]]);
Bc = np.array([[0],
              [1/m]]);
Cc = np.array([1 , 0]);
Dc = np.array([0]);
sys = (Ac,Bc,Cc,Dc);
sysd = signal.cont2discrete(sys,Ts);
(A, B, C, D, Ts) = sysd;
A_T = A.transpose()
C_T = C.reshape(-1,1);

xn = 0.0001; #m^2 position process noise covariance
vn = 0.01; #m^2/s^2 velocity process noise covariance
mn = 0.001; #m^2 measurement noize covariance

R = np.array([[xn, 0],
              [0, vn]]);

Q = np.array([mn]);

pos = t*0;
vel = t*0;
pos_true = t*0;
vel_true = t*0;
k1 = t*0;
k2 = t*0;
cov1 = t*0;
cov2 = t*0;

mew_prev = np.array([[0],
                     [0]]);
ebsilon_prev = np.array([[1,0],
                         [0,1]]);

for i in range(0,len(t)):

    #print(t[i]);
    #prediction
    u = F[i]
    mew_bel = np.dot(A,mew_prev) + np.dot(B,u);
    ebsilon_bel = np.dot(A , np.dot(ebsilon_prev,A_T) ) + R;

    #estimate measurement
    mew_est = mew_bel + np.array([[np.random.randn() * np.sqrt(xn)],
                                  [np.random.randn() * np.sqrt(vn)]]);
    pos_true[i] = mew_est[0];
    vel_true[i] = mew_est[1];
    #z = np.dot(C,mew_bel) + np.sqrt(Q)*np.random.randn();

    #correction
    temp = np.dot(C , np.dot(ebsilon_bel,C_T)) + Q
    if np.shape(temp) == (1,):
        inv = 1/temp;
    else:
        inv = np.linalg.inv(temp);
    K = np.dot(ebsilon_bel , np.dot(C_T,[inv]));
    k1[i] = K[0];
    k2[i] = K[1];
    temp = [z[0][i]] - np.dot(C,mew_bel);
    mew = mew_bel + np.dot(K , [temp]);
    ebsilon = np.dot( (np.identity(2) - np.dot(K,[C])) , ebsilon_bel);
    pos[i] = mew[0];
    vel[i] = mew[1];
    cov1[i] = np.sqrt(ebsilon[0][0]);
    cov2[i] = np.sqrt(ebsilon[1][1]);
    mew_prev = mew;
    ebsilon_prev = ebsilon;

plt.figure(1);
plt.plot(t, pos, label="Estimated Position");
plt.plot(t, pos_true, label="True Position");
plt.ylabel("Position (m)")
plt.xlabel("Time (sec)")
plt.legend();
plt.show();

plt.figure(2);
plt.plot(t, vel, label="Estimated Velocity");
plt.plot(t, vel_true, label="True Velocity");
plt.ylabel("Velocity (m/s)")
plt.xlabel("Time (sec)")
plt.legend();
plt.show();

plt.figure(3);
plt.plot(t,k1,label="K of Position");
plt.plot(t,k2,label="K of Velocity");
plt.ylabel("Kalman Gain")
plt.xlabel("Time (sec)")
plt.legend();
plt.show();

plt.figure(4);
plt.plot(t,cov1,label="Upper Covariance Error");
plt.plot(t,-cov1, label = "Lower Covariance Error");
plt.plot(t,pos-pos_true,label='Position Estimate Error');
plt.xlabel("Time (sec)")
plt.title("Position Error")
plt.legend();
plt.show();

plt.figure(5);
plt.plot(t,cov2,label="Upper Covariance Error");
plt.plot(t,-cov2,label="Lower Covariance Error");
plt.plot(t,vel-vel_true,label='Velocity Estimate Error');
plt.xlabel("Time (sec)")
plt.title("Velocity Error")
plt.legend();
plt.show();
