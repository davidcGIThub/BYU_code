"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
        2/24/2020 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as AP
from tools.transfer_function import transfer_function
from tools.wrap import wrap
from chap6.pid_control import pidControl, piControl, pdControlWithRate
from message_types.msg_state import msg_state

class autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = pdControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=AP.delta_a_max)
        self.course_from_roll = piControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=AP.roll_max)
        self.yaw_damper = transfer_function(
                        num=np.array([[AP.yaw_damper_kp, 0]]),
                        den=np.array([[1, 1/AP.yaw_damper_tau_r]]),
                        Ts=ts_control)

        # instantiate lateral controllers
        self.pitch_from_elevator = pdControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=AP.delta_e_max)
        self.altitude_from_pitch = piControl(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=AP.pitch_max)
        self.airspeed_from_throttle = piControl(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=AP.throttle_max)
        self.commanded_state = msg_state()

    def update(self, cmd, state):

        # lateral autopilot
        chi_c = wrap(cmd.course_command,state.chi) #course command
        phi_c = self.course_from_roll.update(chi_c,state.chi) #roll command
        # print("phi_c")
        # print(phi_c)
        # print("state.phi")
        # print(state.phi)
        # print("state.p")
        # print(state.p)
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p) #aileron command
        # print("delta_a")
        # print(delta_a)

        delta_r = self.yaw_damper.update(state.r) #rudder command

        # longitudinal autopilot
        # saturate the altitude command
        h_c = self.saturate(cmd.altitude_command,AP.altitude_zone[0],AP.altitude_zone[1])
        theta_c = self.altitude_from_pitch.update(h_c,state.h)
        delta_e = self.pitch_from_elevator.update(theta_c,state.theta,state.q)
        Va_command = self.saturate(cmd.airspeed_command,0,60)
        delta_t = self.airspeed_from_throttle.update(Va_command,state.Va)

        # construct output and commanded states
        delta = np.array([[delta_e], [delta_a], [delta_r], [delta_t]])
        self.commanded_state.h = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
