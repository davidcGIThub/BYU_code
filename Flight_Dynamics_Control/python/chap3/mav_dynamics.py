"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
part of mavsimPy
    - Beard & McLain, PUP, 2012
    - Update history:  
        12/17/2018 - RWB
        1/14/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state

import parameters.aerosonde_parameters as MAV
from tools.tools import Quaternion2Euler

class mav_dynamics:
    def __init__(self, Ts):
        self.ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        self._state = np.array([[MAV.pn0], [MAV.pe0], [MAV.pd0], [MAV.u0], [MAV.v0], [MAV.w0], [MAV.e0], [MAV.e1], [MAV.e2], [MAV.e3], [MAV.p0], [MAV.q0], [MAV.r0]])
        self.mass = MAV.mass
        self.Jx = MAV.Jx
        self.Jy = MAV.Jy
        self.Jz = MAV.Jz
        self.Jxz = MAV.Jxz
        self.msg_true_state = msg_state()

    ###################################
    # public functions
    def update_state(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics.
            Inputs are the forces and moments on the aircraft.
            Ts is the time step between function calls.
        '''

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self.ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the message class for the true state
        self._update_msg_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        pn_dot = u*(e1**2+e0**2-e2**2-e3**2) + v*2*(e1*e2-e3*e0) + w*2*(e1*e3+e2*e0)
        pe_dot = u*2*(e1*e2+e3*e0) + v*(e2**2+e0**2-e1**2-e3**2) + w*2*(e2*e3-e1*e0)
        pd_dot = u*2*(e1*e3-e2*e0) + v*2*(e2*e3+e1*e0) + w*(e3**2+e0**2-e1**2-e2**2)

        # position dynamics
        u_dot = r*v - q*w + fx/self.mass
        v_dot = p*w - r*u + fy/self.mass
        w_dot = q*u - p*v + fz/self.mass

        # rotational kinematics
        e0_dot = (-p*e1 - e2*q - e3*r) / 2
        e1_dot = (e0*p + e2*r - e3*q) / 2
        e2_dot = (e0*q - e1*r + e3*p) / 2
        e3_dot = (e0*r + e1*q -e2*p) / 2

        # rotatonal dynamics
        Gamma = self.Jx*self.Jz - self.Jxz**2
        Gamma1 = (self.Jxz*(self.Jx-self.Jy+self.Jz))/Gamma
        Gamma2 = (self.Jz*(self.Jz-self.Jy)+self.Jxz**2)/Gamma
        Gamma3 = self.Jz/Gamma
        Gamma4 = self.Jxz/Gamma
        Gamma5 = (self.Jz-self.Jx)/self.Jy
        Gamma6 = self.Jxz/self.Jy
        Gamma7 = ((self.Jx-self.Jy)*self.Jx+self.Jxz**2)/Gamma
        Gamma8 = self.Jx/Gamma
        p_dot = Gamma1*p*q - Gamma2*q*r + Gamma3*l + Gamma4*n
        q_dot = Gamma5*p*r - Gamma6*(p**2-r**2) + m/self.Jy
        r_dot = Gamma7*p*q - Gamma1*q*r + Gamma4*l + Gamma8*n

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_msg_true_state(self):
        # update the true state message:
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)