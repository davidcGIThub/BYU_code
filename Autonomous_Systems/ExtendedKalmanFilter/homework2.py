#homework 2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from RobotMotion import RobotMotion as robot

t = np.linspace(0,20,20/0.1+1);
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

rb = robot(x0,y0,theta0,alpha1,alpha2,alpha3,alpha4,dt);


fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-10, 10), ylim=(-10, 10))
robot_fig = plt.Polygon(rb.getPoints());
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes);

def init():
    #initialize animation
    global rb
    ax.add_patch(robot_fig);
    time_text.set_text('');
    return robot_fig, time_text

def animate(i):
    global rb, t, vc, wc;
    u = np.array([vc[i],wc[i]]);
    rb.vel_motion_model(u);
    robot_fig.xy  = rb.getPoints();
    time_text.set_text('time = %.1f' % t[i])
    return robot_fig, time_text

from time import time
animate(0);

ani = animation.FuncAnimation(fig, animate, frames = np.size(t), 
                            interval = dt * 1000, blit = True, init_func = init, repeat = False)

plt.show();

#plt.figure(1);
#plt.axis([-10, 10, -10, 10])
#for i in range(0,len(t)):
#    u = np.array([vc[i],wc[i]]);
#    rb.vel_motion_model(u);
#    rb.drawRobot();
#plt.show();
