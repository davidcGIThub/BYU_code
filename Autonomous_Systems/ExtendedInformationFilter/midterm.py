#midterm.py
import numpy as np 
import matplotlib.pyplot as plt 
from QuadCopterMotion import QuadCopterMotion as qcm
from Landmark_Model import LandmarkModel as meas_mod 
from ExtendedInformationFilter import EIF 
import matplotlib.animation as animation
from data_initialization import *

#initialize estimation objects
quad = qcm(x0,y0,theta0,sig_v,sig_w,dt)
quadEst = qcm(x0,y0,theta0,sig_v,sig_w,dt)
measDev = meas_mod(sig_r,sig_phi)
eif = EIF(dt,sig_v,sig_w,sig_r,sig_phi,landmarks)
omega = np.linalg.inv(sigma)
xi = np.dot(omega,mu)

#data save
xi_data = np.zeros((3,np.size(t)))
cov_data = np.zeros((3,np.size(t)))
x_data = np.zeros(np.size(t))
y_data = np.zeros(np.size(t))
theta_data = np.zeros(np.size(t))

#initialize figures
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-15, 15), ylim=(-15, 15))
ax.grid()
quad_fig = plt.Polygon(quad.getPoints(),fc = 'g')
quadEst_fig = plt.Polygon(quadEst.getPoints(),fill=False)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ms = 12 #landmark size
lmd_figs, = ax.plot([],[], 'bo', ms=ms); 
lmdMeas_figs, = ax.plot([],[], 'ko', fillstyle = 'none', ms=ms); 

def init():
    #initialize animation
    ax.add_patch(quad_fig)
    ax.add_patch(quadEst_fig)
    time_text.set_text('0.0')
    lmd_figs.set_data(landmarks[:,0],landmarks[:,1])
    lmdMeas_figs.set_data([],[])
    return quad_fig, time_text, lmd_figs, lmdMeas_figs, quadEst_fig

def animate(i):
    global quad, quadEst, measDev, v, w, t, ms, eif, omega, xi, sigma, mu
    #propogate quadcopter motion
    quad.setState(x_tr[i],y_tr[i],theta_tr[i])
    quad_fig.xy = quad.getPoints()
    state = quad.getState()

    #measure landmark position
    Ranges = range_tr[i,:].flatten().reshape(-1,1)
    Bearings = bearing_tr[i,:].flatten().reshape(-1,1)
    z = np.concatenate((Ranges,Bearings),1)
    landmark_meas = measDev.getLandmarkMeasurements(state,Ranges,Bearings)
    lmdMeas_figs.set_data(landmark_meas[:,0], landmark_meas[:,1])
    lmdMeas_figs.set_markersize(ms)

    #estimate quadcopter position
    u = np.array([v[i],w[i]])
    (xi, omega, mu, sigma) = eif.EIF_localization(xi,omega,mu,sigma,u,z)
    quadEst.setState(mu[0],mu[1],mu[2])
    quadEst_fig.xy = quadEst.getPoints()

    #update time
    time_text.set_text('time = %.1f' % t[i])

    #save data
    xi_data[:,i] = xi
    cov_data[0][i] = sigma[0][0];
    cov_data[1][i] = sigma[1][1];
    cov_data[2][i] = sigma[2][2];
    x_data[i] = mu[0]
    y_data[i] = mu[1]
    theta_data[i] = mu[2]

    return quad_fig, time_text, lmd_figs, lmdMeas_figs, quadEst_fig

from time import time
animate(0);

ani = animation.FuncAnimation(fig, animate, frames = np.size(t), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)

plt.show()


err_bnd_x = 2*np.sqrt(cov_data[0][:])
err_bnd_y = 2*np.sqrt(cov_data[1][:])
err_bnd_th = 2*np.sqrt(cov_data[2][:])

figure1, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(t,x_tr, label = 'true')
ax1.plot(t,x_data, label = 'estimate')
ax1.legend()
ax1.set(ylabel = 'x position (m)')
ax2.plot(t,y_tr)
ax2.plot(t,y_data)
ax2.set(ylabel = 'y position (m)')
ax3.plot(t,theta_tr)
ax3.plot(t,theta_data)
ax3.set(ylabel = 'heading (deg)', xlabel= ("time (s)"))
plt.show()

figure2, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(t,xi_data[0,:])
ax1.set(ylabel = '1')
ax2.plot(t,xi_data[1,:])
ax2.set(ylabel = '2')
ax3.plot(t,xi_data[2,:])
ax3.set(ylabel = '3', xlabel= ("time (s)"))
plt.show()

figure3, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(t,x_tr-x_data, label = 'error', color = 'b')
ax1.plot(t,err_bnd_x, label = 'error_bound', color = 'r')
ax1.plot(t,-err_bnd_x, color = 'r')
ax1.legend()
ax1.set(ylabel = 'x error')
ax2.plot(t,y_tr-y_data, color = 'b')
ax2.plot(t,err_bnd_y, color = 'r')
ax2.plot(t,-err_bnd_y, color = 'r')
ax2.set(ylabel = 'y error (m)')
ax3.plot(t,theta_tr-theta_data,color = 'b')
ax3.plot(t,err_bnd_th,color = 'r')
ax3.plot(t,-err_bnd_th,color = 'r')
ax3.set(ylabel = 'heading error (rad)', xlabel= ("time (s)"))
plt.show()

figure4, (ax1) = plt.subplots(1,1)
ax1.plot(x_data,y_data,label = "estimate" , color = 'r')
ax1.plot(x_tr,y_tr,label = "truth" , color = 'g')
ax1.scatter(landmarks[:,0],landmarks[:,1],color='b')
plt.show()