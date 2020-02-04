"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state

import parameters.aerosonde_parameters as MAV
from tools.tools import Quaternion2Rotation, Quaternion2Euler

class mav_dynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]

        self._state = np.array([[MAV.pn0],  # (0)  # inertial north position
                               [MAV.pe0],   # (1)  # inertial east position
                               [MAV.pd0],   # (2)  # inertial down position, neg of altitude
                               [MAV.u0],    # (3)  # Body frame velocity nose direction (i)
                               [MAV.v0],    # (4)  # Body frame velocity right wing direction (j)
                               [MAV.w0],    # (5)  # Body frame velocity down direction (l)
                                                   # Quaternion rotation from inertial frame to the body frame
                               [MAV.e0],    # (6)  # related to scalar part of rotation = cos(theta/2)
                               [MAV.e1],    # (7)  # related to vector we are rotating about = v*sin(theta/2)
                               [MAV.e2],    # (8)  # " "
                               [MAV.e3],    # (9)  # " "
                               [MAV.p0],    # (10) # roll rate in body frame
                               [MAV.q0],    # (11) # pitch rate in body frame
                               [MAV.r0]])   # (12) # yaw rate in body frame

        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec (velocity of wind [uw, vw, ww])
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]]) #forces acting on mav in body frame [fx,fy,fz]
        self._Va = MAV.u0 # velocity magnitude of airframe relative to airmass
        self._alpha = 0 #angle of attack
        self._beta = 0  #sideslip angle
        # initialize true_state message
        self.msg_true_state = msg_state()

    ###################################
    # public functions
    def update_state(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs (aileron, elevator, rudder, thrust??)
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
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

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_msg_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
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
        u_dot = r*v - q*w + fx/MAV.mass
        v_dot = p*w - r*u + fy/MAV.mass
        w_dot = q*u - p*v + fz/MAV.mass

        # rotational kinematics
        e0_dot = (-p*e1 - e2*q - e3*r) / 2
        e1_dot = (e0*p + e2*r - e3*q) / 2
        e2_dot = (e0*q - e1*r + e3*p) / 2
        e3_dot = (e0*r + e1*q -e2*p) / 2

        # rotatonal dynamics
        Gamma = MAV.Jx*MAV.Jz - MAV.Jxz**2
        Gamma1 = (MAV.Jxz*(MAV.Jx-MAV.Jy+MAV.Jz))/Gamma
        Gamma2 = (MAV.Jz*(MAV.Jz-MAV.Jy)+MAV.Jxz**2)/Gamma
        Gamma3 = MAV.Jz/Gamma
        Gamma4 = MAV.Jxz/Gamma
        Gamma5 = (MAV.Jz-MAV.Jx)/MAV.Jy
        Gamma6 = MAV.Jxz/MAV.Jy
        Gamma7 = ((MAV.Jx-MAV.Jy)*MAV.Jx+MAV.Jxz**2)/Gamma
        Gamma8 = MAV.Jx/Gamma
        p_dot = Gamma1*p*q - Gamma2*q*r + Gamma3*l + Gamma4*n
        q_dot = Gamma5*p*r - Gamma6*(p**2-r**2) + m/MAV.Jy
        r_dot = Gamma7*p*q - Gamma1*q*r + Gamma4*l + Gamma8*n

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        # wind 
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        roll, pitch, yaw = Quaternion2Euler(self._state[6:10])
        # compute airspeed
        u = self._state[3] # Body frame velocity nose direction (i)
        v = self._state[4] # Body frame velocity right wing direction (j)
        w = self._state[5] # Body frame velocity down direction (k)
        Rbv = np.array([[np.cos(pitch)*np.cos(yaw) , np.cos(pitch)*np.sin(yaw) , -np.sin(pitch)], #rotation from vehicle frame to the body frame
                        [np.sin(roll)*np.sin(pitch)*np.cos(yaw)-np.cos(roll)*np.sin(yaw) , np.sin(roll)*np.sin(pitch)*np.sin(yaw)+np.cos(roll)*np.cos(yaw) , np.sin(roll)*np.cos(pitch)],
                        [np.cos(roll)*np.sin(pitch)*np.cos(yaw)+np.sin(roll)*np.sin(yaw) , np.cos(roll)*np.sin(pitch)*np.sin(yaw)-np.sin(roll)*np.cos(yaw) , np.cos(roll)*np.cos(pitch)]])
        Vws = np.dot(Rbv * wind[0:3][:,None]) # ambient wind body frame
        Vwg = wind[3:6][:,None]               # gust wind in body frame
        Vw = Vws + Vwg                        # total wind in body frame
        Vba = np.array([[u],[v],[w]]) - Vw    # airspeed vector (aircraft relative to airmass)
        self._Va = np.linalg.norm(Vba)        # magnitude of airspeed vector
        # compute angle of attack
        ur = Vba[0]
        vr = Vba[1]
        wr = Vba[2]
        self._alpha = np.arctan2(wr,ur)
        # compute sideslip angle
        self._beta = np.arcsin(vr/(self._Va))

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        #forces due to gravity
        Fgx = -
        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz
        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.Vg =
        self.msg_true_state.gamma =
        self.msg_true_state.chi =
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)
