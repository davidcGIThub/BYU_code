#homework 4
#homework #3
import numpy as np 
import matplotlib.pyplot as plt 
from RobotMotion import RobotMotion as rbm 
from LandmarkModel import LandmarkModel as lmm 
from MonteCarloLocalization import MCL
import matplotlib.animation as animation
from parameters import *

#initialize data
x_true = t * 0
y_true = t * 0
theta_true = t * 0
x_est = t * 0
y_est = t * 0
theta_est = t * 0
state = np.array([x0,y0,theta0])
mu = np.array([x0,y0,theta0])
len_m = np.size(m,0)
alpha = np.array([alpha1, alpha2, alpha3, alpha4])

#Initialize Estimation Objects
rb = rbm(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt)
rb_est = rbm(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt)
meas = lmm(sig_r , sig_b)
mcl = MCL(dt,alpha,sig_r,sig_b)
ki_x = np.random.uniform(-10,10,M)
ki_y = np.random.uniform(-10,10,M)
ki_th = np.random.uniform(0,2*np.pi,M)
ki = np.array([ki_x, ki_y, ki_th])

#initialize figures
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-10, 10), ylim=(-10, 10))
ax.grid()
robot_fig = plt.Polygon(rb.getPoints(),fc = 'g')
robot_est_fig = plt.Polygon(rb_est.getPoints(),fill=False)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ms = 12
lmd_figs, = ax.plot([],[], 'bo', ms=ms)
lmd_meas_figs, = ax.plot([],[], 'ko', fillstyle = 'none', ms=ms)

def init():
    #initialize animation
    ax.add_patch(robot_fig)
    ax.add_patch(robot_est_fig)
    time_text.set_text('0.0')
    lmd_figs.set_data(m[:,0],m[:,1])
    lmd_meas_figs.set_data([],[])
    return robot_fig, robot_est_fig, time_text, lmd_figs, lmd_meas_figs

def animate(i):
    global rb, rb_est,ki, meas, t, vc, wc, mu, ms
    #propogate robot motion
    u = np.array([vc[i],wc[i]])
    rb.vel_motion_model(u)
    robot_fig.xy  = rb.getPoints()
    state = rb.getState()
    #measure landmark position
    if cycle:
        m_temp = np.array([m[i%len_m]])
    else:
        m_temp = m
    landmarks_meas = meas.getLandmarks(state,m_temp)
    Ranges = meas.getRanges(state,m_temp)
    Bearings = meas.getBearings(state,m_temp)
    z = np.array([Ranges.flatten(), Bearings.flatten()])
    lmd_meas_figs.set_data(landmarks_meas[:,0], landmarks_meas[:,1])
    lmd_meas_figs.set_markersize(ms)
    #estimate robot motion
    (ki, mu)  = mcl.MCL_Localization(ki,u,z,m_temp)
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

    return robot_fig, robot_est_fig, time_text, lmd_figs, lmd_meas_figs

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(t), 
                            interval = dt*animation_speed, blit = True, init_func = init, repeat = False)

plt.show()
