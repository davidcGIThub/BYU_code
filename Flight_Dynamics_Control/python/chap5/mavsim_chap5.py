"""
mavsim_python
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/2/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM
import parameters.aerosonde_parameters as MAV

from chap4.mav_viewer import mav_viewer
from chap4.data_viewer import data_viewer
from chap4.mav_dynamics import mav_dynamics
from chap4.wind_simulation import wind_simulation
from chap5.trim import compute_trim
#from chap5.compute_models import compute_model
from tools.signals import signals

# initialize the visualization
mav_view = mav_viewer()  # initialize the mav viewer
#data_view = data_viewer()  # initialize view of data plots

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)

# use compute_trim function to compute trim state and trim input
Va0 = np.sqrt(MAV.u0**2 + MAV.v0**2 + MAV.w0**2)
gamma = 10.*np.pi/180.
delta0 = np.array([[0],[0],[0],[0.5]])  #   [delta_e, delta_a, delta_r, delta_t]
trim_state, trim_input = compute_trim(mav, delta0, Va0, gamma)
mav._state = trim_state  # set the initial state of the mav to the trim state
delta = trim_input  # set input to constant trim input

# # compute the state space model linearized about trim
#compute_model(mav, trim_state, trim_input)

# this signal will be used to excite modes
#input_signal = signals(amplitude=.05,
#                       duration=0.01,
#                       start_time=2.0)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    # -------physical system-------------
    current_wind = np.zeros((6,1))
    # this input excites the phugoid mode by adding an impulse at t=5.0
    # delta[0][0] += input_signal.impulse(sim_time)
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    #data_view.update(mav.true_state,  # true states
    #                 mav.true_state,   # estimated states
    #                 mav.true_state,  # commanded states
    #                 SIM.ts_simulation)

    # -------increment time-------------
    sim_time += SIM.ts_simulation




