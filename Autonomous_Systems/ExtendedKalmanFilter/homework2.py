#homework 2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from RobotMotion import RobotMotion as robot
from LandmarkModel import LandmarkModel as lmd

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
alpha1 = 0.1;
alpha2 = 0.01;
alpha3 = 0.01;
alpha4 = 0.1;
step = 0;

std_r = 0.1;
std_b = 0.05;

rb = robot(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt);
landmarks = lmd(np.array([[6,4],[-7,8],[6,-4]]),std_r, std_b)

'''
plt.figure(1);
true_loc = landmarks.getLocations();
est_loc = landmarks.estimateLocations(np.array([-5,-3]));
plt.plot(true_loc[:,0], true_loc[:,1],'-.or');
plt.plot(est_loc[:,0], est_loc[:,1],'-.ob');
plt.show();
'''

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-10, 10), ylim=(-10, 10));
ax.grid();
robot_fig = plt.Polygon(rb.getPoints(),fc = 'g');
locations = landmarks.getLocations();
lmd1 = plt.Circle(locations[0], radius = 0.5, fc = 'b');
lmd2 = plt.Circle(locations[1], radius = 0.5, fc = 'b');
lmd3 = plt.Circle(locations[2], radius = 0.5, fc = 'b');
estimated_locations = landmarks.estimateLocations(np.array([-5,-3]))
lmd1_est = plt.Circle(estimated_locations[0], radius = 0.5, fill = False)
lmd2_est = plt.Circle(estimated_locations[1], radius = 0.5, fill = False)
lmd3_est = plt.Circle(estimated_locations[2], radius = 0.5, fill = False)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes);

def init():
    #initialize animation
    ax.add_patch(robot_fig);
    ax.add_patch(lmd1);
    ax.add_patch(lmd2);
    ax.add_patch(lmd3);
    ax.add_patch(lmd1_est);
    ax.add_patch(lmd2_est);
    ax.add_patch(lmd3_est);
    time_text.set_text('');
    return robot_fig, lmd1_est, lmd2_est, lmd3_est, time_text

def animate(i):
    global rb, landmarks, t, vc, wc;
    #propogate robot motion
    u = np.array([vc[i],wc[i]]);
    rb.vel_motion_model(u);
    robot_fig.xy  = rb.getPoints();
    state = rb.getState();
    #estimate landmark position
    est_locs = landmarks.estimateLocations(np.array([state[0],state[1]]))
    lmd1_est.center = est_locs[0];
    lmd2_est.center = est_locs[1];
    lmd3_est.center = est_locs[2];

    #update time
    time_text.set_text('time = %.1f' % t[i])
    #save state information
    x_true = state[0];
    y_true = state[1];
    theta_true = state[2];

    return robot_fig, lmd1_est, lmd2_est, lmd3_est, time_text

from time import time
animate(0);

ani = animation.FuncAnimation(fig, animate, frames = np.size(t), 
                            interval = dt * 1000, blit = True, init_func = init, repeat = False)

plt.show();