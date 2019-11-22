#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from RobotMotion import RobotMotion as robot
from MeasurementModel import MeasurementModel as mmd
from FastSLAM import Fast_SLAM as FS
from data_initialization import *


rb = robot(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt)
rb_est = robot(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt)
measDevice = mmd(sig_r,sig_b)
fs = FS(dt,alpha,sig_r,sig_b,pose_noise)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-x_limits,x_limits), ylim=(-y_limits,y_limits))
ax.grid()
robot_fig = plt.Polygon(rb.getPoints(),fc = 'g')

robot_est_fig = plt.Polygon(rb_est.getPoints(),fill=False)
lmd_figs, = ax.plot([],[], 'bo', ms=ms); 
lmdMeas_figs, = ax.plot([],[], 'ko', fillstyle = 'none', ms=ms)
cov_figs, =  ax.plot([],[], '.', ms = .1)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
particles, = ax.plot([], [], 'ko', ms=1)

def init():
    #initialize animation
    ax.add_patch(robot_fig)
    ax.add_patch(robot_est_fig)
    lmd_figs.set_data(landmarks[:,0],landmarks[:,1])
    lmdMeas_figs.set_data([],[])
    #cov_figs.set_data([],[])
    time_text.set_text('')
    particles.set_data([], [])
    return robot_fig, robot_est_fig, cov_figs, lmd_figs, lmdMeas_figs, time_text, particles

def animate(i):
    global rb, rb_est, landmarks, t, vc, wc, pose, Sig, c , detected_flag, Y, features
    #propogate robot motion
    u = np.array([vc[i],wc[i]])
    rb.vel_motion_model(u)
    robot_fig.xy  = rb.getPoints()
    state = rb.getState()
    #measure landmark position
    Ranges = measDevice.getRanges(state,landmarks)
    (Bearings,c) = measDevice.getBearings(state,landmarks,fov)
    z = np.concatenate((Ranges,Bearings),1)
    #estimate robot motion
    Y = fs.fast_SLAM_1(z, c, u, Y, detected_flag)
    x_ave = np.mean(Y[:,0])
    y_ave = np.mean(Y[:,1])
    th_ave = np.mean(Y[:,2])
    pose = np.array([x_ave,y_ave,th_ave])
    rb_est.setState(pose[0],pose[1],pose[2])
    #adjust flags
    detected_flag[c > 0] = 1
    robot_est_fig.xy = rb_est.getPoints()
    #update landmark estimates
    lmdMeas_figs.set_data(features[:,0], features[:,1])
    lmdMeas_figs.set_markersize(ms)
    x_particles = Y[:,0]
    y_particles = Y[:,1]
    temp_cov = np.zeros(2*N)
    for k in range(0,N):
        features[k,0] = np.mean(Y[:,3+k*6])
        features[k,1] = np.mean(Y[:,4+k*6])
        x_particles = np.append(x_particles,Y[:,3+k*6])
        y_particles = np.append(y_particles,Y[:,4+k*6])
        temp_cov[2*k] = np.mean(Y[:,5+k*6])
        temp_cov[2*k+1] = np.mean(Y[:,8+k*6])
    particles_data = np.concatenate((x_particles[:,None],y_particles[:,None]),1)
    #particles
    if(show_particles):
        particles.set_data(particles_data[:, 0], particles_data[:, 1])
        particles.set_markersize(1)
    #plot covariance bounds
    cov[:,i] = temp_cov
    points = measDevice.getCovariancePoints(features,cov[:,i])
    cov_figs.set_data(points[:,0], points[:,1])
    cov_figs.set_markersize(.5)
    #update time
    time_text.set_text('time = %.1f' % t[i])
    #save state information
    x_true[i] = state[0]
    y_true[i] = state[1]
    theta_true[i] = state[2]
    x_est[i] = pose[0]
    y_est[i] = pose[1]
    theta_est[i] = pose[2]
    return robot_fig, robot_est_fig, time_text, lmd_figs, lmdMeas_figs, particles, cov_figs

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(t), 
                            interval = dt*1000, blit = True, init_func = init, repeat = False)

plt.show()


#err_bnd_x = 2*np.sqrt(cov[0][:])
#err_bnd_y = 2*np.sqrt(cov[1][:])
#err_bnd_th = 2*np.sqrt(cov[2][:])

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
plt.show()
'''
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
heading_diff = theta_true-theta_est
heading_diff -= np.pi * 2 * np.floor((heading_diff + np.pi) / (2 * np.pi))
ax3.plot(t,heading_diff,color = 'b')
ax3.plot(t,err_bnd_th,color = 'r')
ax3.plot(t,-err_bnd_th,color = 'r')
ax3.set(ylabel = 'heading error (rad)', xlabel= ("time (s)"))
plt.show()

figure3, (ax1, ax2) = plt.subplots(2,1)
length = np.size(t)
for i in range(0,N):
    ax1.plot(t[3:length],cov[3+2*i,3:length],label = i)
ax1.legend()
ax1.set(ylabel = 'covariance x_landmark pos')
for i in range(0,N):
    ax2.plot(t[3:length],cov[4+2*i,3:length],label = i)
ax2.legend()
ax2.set(ylabel = 'covariance y_landmark pos')
plt.show()

fig4 = plt.figure()
plt.imshow(Sig)
plt.show()
'''