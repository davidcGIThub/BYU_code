#midterm
import numpy as np 
import matplotlib.pyplot as plt
from QuadCopterMotion import QuadCoptorMotion as qcm
from LandmarkModel import LandmarkModel as lmm 
from ExtendedInformationFilter import EIF 
import matplotlib.animation as animation

#initialize estimation objects
quad = qcm(x0,y0,theta0,sig_v,sig_w,dt)
quadEst = qcm(x0,y0,theta0,sig_v,sig_w,dt)
measDev = lmm(landmarks,sig_r,sig_phi)
eif = EIF()

#initialize figures
quad_fig = plt.Polygon(quad.getPoints(),fc = 'g')
quadEst_fig = plt.Polygon(quadEst_fig.getPoints(),fill=False)
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
    return quad_fig, quadEst_fig, time_text, lmd_figs, lmdMeas_figs

def animate(i):
    global quad, quadEst, measDev, eif, vc, wc
    #propogate quadcopter motion
    u = np.array([vc[i]],wc[i]])
    quad.vel_motion_model(u)
    
