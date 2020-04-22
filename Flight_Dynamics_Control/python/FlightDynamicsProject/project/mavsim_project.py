"""
mavsim_python
Full State EKF Implementation
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from project.mav_viewer import mav_viewer
from project.data_viewer import data_viewer
from project.data_viewer2 import data_viewer2
from project.wind_simulation import wind_simulation
from project.autopilot import autopilot
from project.mav_dynamics import mav_dynamics
from project.observer import observer
from project.fullStateDirectObserver import fullStateDirectObserver as fsdObserver 
from project.fullStateIndirectObserver import fullStateIndirectObserver as fsiObserver
from tools.signals import signals
from timeit import default_timer as timer

observer_number = 2
compare_estimators = False
N = 1 #loop iteration
ave_time = 0

# initialize the visualization
mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots
if compare_estimators:
    data_view2 = data_viewer2()  # initialize view of data plots


# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)
ctrl = autopilot(SIM.ts_simulation)
obsv = observer(SIM.ts_simulation)
obsv1 = fsdObserver(SIM.ts_simulation)
obsv2 = fsiObserver(SIM.ts_simulation)


# autopilot commands
from message_types.msg_autopilot import msg_autopilot
commands = msg_autopilot()
Va_command = signals(dc_offset=25.0,
                     amplitude=3.0,
                     start_time=2.0,
                     frequency = 0.01)
h_command = signals(dc_offset=100.0,
                    amplitude=10.0,
                    start_time=0.0,
                    frequency=0.02)
chi_command = signals(dc_offset=np.radians(180),
                      amplitude=np.radians(45),
                      start_time=5.0,
                      frequency=0.015)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    # -------autopilot commands-------------
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = chi_command.square(sim_time)
    commands.altitude_command = h_command.square(sim_time)

    start = timer()
    # ------- estimator -----------
    measurements = mav.sensors()  # get sensor measurements
    if observer_number == 0 or compare_estimators:
        EKF_estimate = obsv.update(measurements)
        estimate = EKF_estimate
    if observer_number == 1 or compare_estimators:
        FSD_EKF_estimate = obsv1.update(measurements)
        estimate = FSD_EKF_estimate
    if observer_number == 2 or compare_estimators:
        FSI_EKF_estimate = obsv2.update(measurements)
        estimate = FSI_EKF_estimate
    if compare_estimators:
        estimate = mav.msg_true_state
    elapsed_time = timer() - start
    ave_time = elapsed_time/N + ave_time*(N-1)/N
    N += 1
    print("ave_elapsed_time: " , ave_time)

     # -------controller------------
    delta, commanded_state = ctrl.update(commands, estimate) #mav.msg_true_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV

    #update data viewer
    data_view.update(mav.msg_true_state,  # true states
                     estimate,  # estimated states
                     commanded_state,  # commanded states
                     SIM.ts_simulation)
    if compare_estimators:
        data_view2.update(mav.msg_true_state,  # true states
                        EKF_estimate,  # estimated states
                        FSD_EKF_estimate,  # commanded states
                        FSI_EKF_estimate,
                        SIM.ts_simulation)

    # -------increment time-------------
    sim_time += SIM.ts_simulation





