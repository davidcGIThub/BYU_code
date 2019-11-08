#homework #3
import numpy as np 
import matplotlib.pyplot as plt 
from RobotMotion import RobotMotion as rbm 
from LandmarkModel import LandmarkModel as lmm 
from UnscentedKalmanFilter import UKF 
import matplotlib.animation as animation
from givenParameters import *

#initialize data
x_true = t * 0
y_true = t * 0
theta_true = t * 0
x_est = t * 0
y_est = t * 0
theta_est = t * 0
Kalman = np.zeros((3,2,np.size(t)));
cov = np.zeros((3,np.size(t)))
state = np.array([x0,y0,theta0])
mu = np.array([x0,y0,theta0])
Sig = Sig0
len_m = np.size(m,0)
alpha = np.array([alpha1, alpha2, alpha3, alpha4])

#Initialize Estimation Objects
rb = rbm(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt)
rb_est = rbm(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt)
meas = lmm(sig_r , sig_b)
ukf = UKF(dt,alpha,sig_r,sig_b,alfa,kappa,beta)

#initialize figures
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-10, 10), ylim=(-10, 10))
ax.grid()
robot_fig = plt.Polygon(rb.getPoints(),fc = 'g')
robot_est_fig = plt.Polygon(rb_est.getPoints(),fill=False)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ms = 12
lmd_figs, = ax.plot([],[], 'bo', ms=ms); 
lmd_meas_figs, = ax.plot([],[], 'ko', fillstyle = 'none', ms=ms); 

def init():
    #initialize animation
    ax.add_patch(robot_fig)
    ax.add_patch(robot_est_fig)
    time_text.set_text('0.0')
    lmd_figs.set_data(m[:,0],m[:,1])
    lmd_meas_figs.set_data([],[])
    return robot_fig, robot_est_fig, time_text, lmd_figs, lmd_meas_figs

def animate(i):
    global rb, rb_est, meas, t, vc, wc, mu, Sig, ms
    #propogate robot motion
    u = np.array([vc[i],wc[i]])
    rb.vel_motion_model(u)
    robot_fig.xy  = rb.getPoints()
    state = rb.getState()
    #measure landmark position
    if cycle:
        m_temp = np.array([m[i%len_m]]);
    else:
        m_temp = m;
    landmarks_meas = meas.getLandmarks(state,m_temp)
    Ranges = meas.getRanges(state,m_temp)
    Bearings = meas.getBearings(state,m_temp)
    z = np.array([Ranges.flatten(), Bearings.flatten()])
    lmd_meas_figs.set_data(landmarks_meas[:,0], landmarks_meas[:,1])
    lmd_meas_figs.set_markersize(ms)
    #estimate robot motion
    (mu, Sig, K)  = ukf.UKF_Localization(mu,Sig,u,z,m_temp)
    rb_est.setState(mu[0],mu[1],mu[2])
    robot_est_fig.xy  = rb_est.getPoints()
    #update time
    time_text.set_text('time = %.1f' % t[i])
    #save state information
    x_true[i] = state[0]
    y_true[i] = state[1]
    theta_true[i] = state[2]
    x_est[i] = mu[0]
    y_est[i] = mu[1]
    theta_est[i] = mu[2]
    cov[0][i] = Sig[0][0]
    cov[1][i] = Sig[1][1]
    cov[2][i] = Sig[2][2]
    Kalman[:,:,i] = K

    return robot_fig, robot_est_fig, time_text, lmd_figs, lmd_meas_figs

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(t), 
                            interval = dt*animation_speed, blit = True, init_func = init, repeat = False)

plt.show()

err_bnd_x = 2*np.sqrt(cov[0][:])
err_bnd_y = 2*np.sqrt(cov[1][:])
err_bnd_th = 2*np.sqrt(cov[2][:])

figure1, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(t,x_true, label = 'true')
ax1.plot(t,x_est, label = 'estimate')
ax1.legend()
ax1.set(ylabel = 'x position (m)')
ax2.plot(t,y_true)
ax2.plot(t,y_est)
ax2.set(ylabel = 'y position (m)')
ax3.plot(t,theta_true)
ax3.plot(t,theta_est)
ax3.set(ylabel = 'heading (deg)', xlabel= ("time (s)"))
if given:
    ax1.plot(t,x_given, label = 'given')
    ax1.legend()
    ax2.plot(t,y_given)
    ax3.plot(t,theta_given)
plt.show()

figure2, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(t,x_true-x_est, label = 'error', color = 'b')
ax1.plot(t,err_bnd_x, label = 'error_bound', color = 'r')
ax1.plot(t,-err_bnd_x, color = 'r')
ax1.legend()
ax1.set(ylabel = 'x error')
ax2.plot(t,y_true-y_est, color = 'b')
ax2.plot(t,err_bnd_y, color = 'r')
ax2.plot(t,-err_bnd_y, color = 'r')
ax2.set(ylabel = 'y error (m)')
ax3.plot(t,theta_true-theta_est,color = 'b')
ax3.plot(t,err_bnd_th,color = 'r')
ax3.plot(t,-err_bnd_th,color = 'r')
ax3.set(ylabel = 'heading error (rad)', xlabel= ("time (s)"))
plt.show()

figure3, ax1 = plt.subplots(1,1)
ax1.plot(t,Kalman[0,0,:])
ax1.plot(t,Kalman[1,0,:])
ax1.plot(t,Kalman[2,0,:])
ax1.plot(t,Kalman[0,1,:])
ax1.plot(t,Kalman[1,1,:])
ax1.plot(t,Kalman[2,1,:])
ax1.set(ylabel = 'Kalman Gain', xlabel= ("time (s)"))
plt.show()