#midterm
import numpy as np 
import matplotlib.pyplot as plt 
from QuadCopterMotion import QuadCopterMotion as qcm
from LandmarkModel import LandmarkModel as meas_mod 
from ExtendedInformationFilter import EIF 
import matplotlib.animation as animation
from data_initialization import *

#initialize estimation objects
quad = qcm(x0,y0,theta0,sig_v,sig_w,dt)
#quadEst = qcm(x0,y0,theta0,sig_v,sig_w,dt)
measDev = meas_mod(sig_r,sig_phi)
#eif = EIF()

#initialize figures
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-15, 15), ylim=(-15, 15))
ax.grid()
quad_fig = plt.Polygon(quad.getPoints(),fc = 'g')
#quadEst_fig = plt.Polygon(quadEst.getPoints(),fill=False)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ms = 12 #landmark size
lmd_figs, = ax.plot([],[], 'bo', ms=ms); 
lmdMeas_figs, = ax.plot([],[], 'ko', fillstyle = 'none', ms=ms); 

def init():
    #initialize animation
    ax.add_patch(quad_fig)
    #ax.add_patch(quadEst_fig)
    time_text.set_text('0.0')
    lmd_figs.set_data(landmarks[:,0],landmarks[:,1])
    lmdMeas_figs.set_data([],[])
    return quad_fig, time_text, lmd_figs, lmdMeas_figs, #quadEst_fig

def animate(i):
    global quad, quadEst, measDev, v, w, t, ms #eif
    #propogate quadcopter motion
    #u = np.array([v[i]],w[i]])
    quad.setState(x_tr[i],y_tr[i],theta_tr[i])
    quad_fig.xy = quad.getPoints()
    state = quad.getState()

    #measure landmark position
    landmark_meas = measDev.getLandmarkEstimates(state,landmarks)
    #Ranges = meas.getRanges(state,m)
    #Bearings = meas.getBearings(state,m)
    #z = np.array([Ranges.flatten(), Bearings.flatten()])
    lmdMeas_figs.set_data(landmark_meas[:,0], landmark_meas[:,1])
    lmdMeas_figs.set_markersize(ms)

    #update time
    time_text.set_text('time = %.1f' % t[i])

    return quad_fig, time_text, lmd_figs, lmdMeas_figs

from time import time
animate(0);

ani = animation.FuncAnimation(fig, animate, frames = np.size(t), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)

plt.show();
