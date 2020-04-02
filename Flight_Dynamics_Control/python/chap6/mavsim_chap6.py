"""
mavsim_python
    - Chapter 6 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/5/2019 - RWB
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap4.mav_viewer import mav_viewer
from chap4.data_viewer import data_viewer
from chap4.mav_dynamics import mav_dynamics
from chap4.wind_simulation import wind_simulation
from chap6.autopilot import autopilot
from tools.signals import signals
from message_types.msg_autopilot import msg_autopilot
from chap5.trim import compute_trim

# initialize the visualization
mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)
ctrl = autopilot(SIM.ts_simulation)

# autopilot commands
commands = msg_autopilot()
# Va_command = signals(dc_offset=25.0,
#                      amplitude=3.0,
#                      start_time=2.0,
#                      frequency=0.01)
# h_command = signals(dc_offset=100.0,
#                     amplitude=10.0,
#                     start_time=0.0,
#                     frequency=0.02)
# chi_command = signals(dc_offset=np.radians(180),
#                       amplitude=np.radians(45),
#                       start_time=5.0,
#                       frequency=0.015)

Va_command = signals(dc_offset=25.0, amplitude=10.0, start_time=2.0, frequency = 0.02)
h_command = signals(dc_offset=100.0, amplitude=20.0, start_time=0.0, frequency = 0.02)
chi_command = signals(dc_offset=np.radians(0), amplitude=np.radians(200), start_time=5.0, frequency = 0.02)                     

# initialize the simulation time
sim_time = SIM.start_time

#Make initial trim conditions
Va0 = 25
gamma = 0#10.0*np.pi/180.
delta0 = np.array([[0],[0],[0],[0.5]])  #   [delta_e, delta_a, delta_r, delta_t]
trim_state, trim_input = compute_trim(mav, delta0, Va0, gamma)
mav.set_state(trim_state)
flag = False
# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    # -------autopilot commands-------------
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = chi_command.square(sim_time)
    commands.altitude_command = h_command.square(sim_time)

    # -------controller-------------
    estimated_state = mav.msg_true_state  # uses true states in the control
    if(np.abs(estimated_state.chi - 1.57) < .1):
        flag = True
    if(flag == True):
        commands.course_command = -2.5
    delta, commanded_state = ctrl.update(commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    data_view.update(mav.msg_true_state, # true states
                     estimated_state, # estimated states
                     commanded_state, # commanded states
                     SIM.ts_simulation)

    # -------increment time-------------
    sim_time += SIM.ts_simulation




