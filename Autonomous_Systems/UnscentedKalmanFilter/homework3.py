#homework #3
import numpy as np 
import matplotlib.pyplot as plt 
from RobotMotion import RobotMotion as rbm 
from LandmarkModel import LandmarkModel as lmm 
from UnscentedKalmanFilter import UKF 
import matplotlib.animation as animation

#initialize data
t = np.linspace(0,20,20/0.1+1);
x_true = t * 0;
y_true = t * 0;
theta_true = t * 0;
x_est = t * 0;
y_est = t * 0;
theta_est = t * 0;

dt = 0.1;
vc = 1 + 0.5*np.cos(2.0*np.pi*.2*t);
wc = -0.2 + 2*np.cos(2*np.pi*0.6*t);
x0 = -5.0; #m
y0 = -3.0; #m
theta0 = np.pi/2.0; #rad
state = np.array([x0,y0,theta0])
mu = np.array([x0,y0,theta0]);
alpha1 = 0.1;
alpha2 = 0.01;
alpha3 = 0.01;
alpha4 = 0.1;
alpha = np.array([alpha1, alpha2, alpha3, alpha4]);
alfa = 0.5;
kappa = 3.0;
beta = 2;
sig_r = 0.1;
sig_b = 0.05;
Sig = np.array([[1,0,0],
               [0,1,0],
               [0,0,0.1]]);
landmarks = np.array([[6,4]]);#,[-7,8],[6,-4]]);
len_m = np.size(landmarks,0);
#Initialize Objects
rb = rbm(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt);
rb_est = rbm(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt);
m = lmm(landmarks , sig_r , sig_b);
ukf = UKF(dt,alpha,sig_r,sig_b,alfa,kappa,beta);
#initialize figures
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-10, 10), ylim=(-10, 10));
ax.grid();
robot_fig = plt.Polygon(rb.getPoints(),fc = 'g');
robot_est_fig = plt.Polygon(rb_est.getPoints(),fill=False);
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes);
lmd_fig1 = plt.Circle(landmarks[0], radius = 0.5, fc = 'b');
#lmd_fig2 = plt.Circle(landmarks[1], radius = 0.5, fc = 'b');
#lmd_fig3 = plt.Circle(landmarks[2], radius = 0.5, fc = 'b');

def init():
    #initialize animation
    ax.add_patch(robot_fig);
    ax.add_patch(robot_est_fig);
    time_text.set_text('0.0');
    ax.add_patch(lmd_fig1);
    #ax.add_patch(lmd_fig2);
    #ax.add_patch(lmd_fig3);
    return robot_fig, robot_est_fig, time_text, lmd_fig1#, lmd_fig2, lmd_fig3

def animate(i):
    global rb, rb_est, m, t, vc, wc, mu, Sig
    #propogate robot motion
    u = np.array([vc[i],wc[i]]);
    rb.vel_motion_model(u);
    robot_fig.xy  = rb.getPoints();
    state = rb.getState();
    #measure landmark position
    #landmark_pos = m.getLandmarks(state);
    Ranges = m.getRanges(state);
    Bearings = m.getBearings(state);
    z = np.array([Ranges.flatten(), Bearings.flatten()]);
    #estimate robot motion
    #(mu, Sig, landmarks_meas)  = ekf.EKF_Localization(mu,Sig,u,state);
    #rb_est.setState(mu[0],mu[1],mu[2]);
    robot_est_fig.xy  = rb.getPoints();
    #update time
    time_text.set_text('time = %.1f' % t[i])
    #save state information
    x_true[i] = state[0];
    y_true[i] = state[1];
    theta_true[i] = state[2];
    x_est[i] = mu[0];
    y_est[i] = mu[1];
    theta_est[i] = mu[2];
    return robot_fig, robot_est_fig, time_text, lmd_fig1#, lmd_fig2, lmd_fig3

from time import time
animate(0);

ani = animation.FuncAnimation(fig, animate, frames = np.size(t), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)

plt.show();