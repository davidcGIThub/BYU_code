"""
mavsimPy
    - Chapter 3 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/18/2018 - RWB
        1/14/2019 - RWB
"""
import sys
sys.path.append('/home/david/BYU_code/Flight_Dynamics_Control/python/')
import numpy as np
import parameters.simulation_parameters as SIM

from chap3.mav_viewer import mav_viewer
from chap3.data_viewer import data_viewer
from chap3.mav_dynamics import mav_dynamics


# initialize the visualization

mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots

# initialize elements of the architecture
mav = mav_dynamics(SIM.ts_simulation)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time: 
    #-------vary forces and moments to check dynamics-------------
    fx = 0 #10
    fy = 0 # 10
    fz = 0 # 10
    Mx = 0 # 0.1
    My = 0 # 0.1
    Mz = 0 # 0.1
    forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T

    #-------physical system-------------
    mav.update_state(forces_moments)  # propagate the MAV dynamics
    #-------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     mav.msg_true_state, # commanded states
                     SIM.ts_simulation)
    #-------increment time-------------
    sim_time += SIM.ts_simulation




