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
from tools.tools import Quaternion2Euler, Quaternion2Euler

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

        #rotation from vehicle to body
        self.Rbv = np.array([[1,0,0],[0,1,0],[0,0,1]]) #rotation from vehicle to body
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec (velocity of wind [uw, vw, ww])
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]]) #forces acting on mav in body frame [fx,fy,fz]
        self._Va = MAV.u0 # velocity magnitude of airframe relative to airmass
        self._alpha = 0 #angle of attack
        self._beta = 0  #sideslip angle
        self.phi = 0 # roll
        self.theta = 0 # pitch
        self.psi = 0 #yaw
        self.Vg = MAV.u0
        self.gamma = 0 #flight path angle (pitch up from horizontal velocity)
        self.chi = 0 #course angle (heading)
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

        #update Euler States
        temp1, temp2, temp3 = Quaternion2Euler(self._state[6:10])
        self.phi = temp1[0]
        self.theta = temp2[0]
        self.psi = temp3[0]
        #rotation from vehicle to body frame

        self.Rbv = np.array([[np.cos(self.theta)*np.cos(self.psi) ,  #rotation from vehicle frame to the body frame
                            np.cos(self.theta)*np.sin(self.psi) , 
                            -np.sin(self.theta)], 
                        [np.sin(self.phi)*np.sin(self.theta)*np.cos(self.psi)-np.cos(self.phi)*np.sin(self.psi) , 
                            np.sin(self.phi)*np.sin(self.theta)*np.sin(self.psi)+np.cos(self.phi)*np.cos(self.psi) , 
                            np.sin(self.phi)*np.cos(self.theta)],
                        [np.cos(self.phi)*np.sin(self.theta)*np.cos(self.psi)+np.sin(self.phi)*np.sin(self.psi) , 
                            np.cos(self.phi)*np.sin(self.theta)*np.sin(self.psi)-np.sin(self.phi)*np.cos(self.psi) , 
                            np.cos(self.phi)*np.cos(self.theta)]])
        #update inertial velocity
        Va = np.array([self._state[3][0],self._state[4][0],self._state[5][0]])[:,None]
        Vg = np.dot(self.Rbv.T,Va)
        self.Vg = np.linalg.norm(Vg)
        #update heading
        north = np.array([1,0,0])
        Vg_ij_plane = np.array([ Vg[0][0], Vg[1][0] , 0 ])
        self.gamma = np.arccos( np.dot(north,Vg_ij_plane[:None]) / (np.linalg.norm(Vg_ij_plane) * np.linalg.norm(north) ) )
        self.chi = np.arccos( np.dot(Vg_ij_plane, Vg) / (np.linalg.norm(Vg_ij_plane) * np.linalg.norm(Vg) ) )

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
        p_dot = MAV.gamma1*p*q - MAV.gamma2*q*r + MAV.gamma3*l + MAV.gamma4*n
        q_dot = MAV.gamma5*p*r - MAV.gamma6*(p**2-r**2) + m/MAV.Jy
        r_dot = MAV.gamma7*p*q - MAV.gamma1*q*r + MAV.gamma4*l + MAV.gamma8*n

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        # wind 
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        
        # compute airspeed
        u = self._state[3] # Body frame velocity nose direction (i)
        v = self._state[4] # Body frame velocity right wing direction (j)
        w = self._state[5] # Body frame velocity down direction (k)
        Vws = np.dot(self.Rbv , wind[0:3]) # ambient wind body frame
        Vwg = wind[3:6][:,None]               # gust wind in body frame
        Vw = Vws + Vwg                        # total wind in body frame
        Vba = np.array([[u],[v],[w]]) - Vw    # airspeed vector (aircraft relative to airmass)
        self._Va = np.linalg.norm(Vba)        # magnitude of airspeed vector
        # compute angle of attack
        ur = Vba[0][0]
        vr = Vba[1][0]
        wr = Vba[2][0]
        self._alpha = np.arctan2(wr,ur)[0]
        # compute sideslip angle
        self._beta = np.arcsin(vr/(self._Va))

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        delta_a = delta[0,0]
        delta_e = delta[1,0]
        delta_r = delta[2,0]
        delta_t = delta[3,0]

        #forces due to gravity
        Fgx = -MAV.mass*MAV.gravity*np.sin(self.theta)
        Fgy = MAV.mass*MAV.gravity*np.cos(self.theta)*np.sin(self.phi)
        Fgz = MAV.mass*MAV.gravity*np.cos(self.theta)*np.cos(self.phi)
        #forces due to air
        p = self._state[10][0]
        q = self._state[11][0]
        r = self._state[12][0]
        sigma_alpha = ( 1 + np.exp(-MAV.M*(self._alpha-MAV.alpha0)) + np.exp(MAV.M*(self._alpha+MAV.alpha0)) ) \
            /( (1+np.exp(-MAV.M*(self._alpha-MAV.alpha0))) * (1+np.exp(MAV.M*(self._alpha+MAV.alpha0)))  ) 
        Cl_alpha = (1-sigma_alpha) * (MAV.C_L_0 + MAV.C_L_alpha*self._alpha) + \
            sigma_alpha*(2*np.sign(self._alpha)*np.cos(self._alpha)*np.sin(self._alpha)**2)
        Cd_alpha = MAV.C_D_p + ((MAV.C_L_0 + MAV.C_L_alpha*self._alpha)**2) / (np.pi*MAV.e*MAV.AR)
        Cx_alpha = -Cd_alpha*np.cos(self._alpha) + Cl_alpha*np.sin(self._alpha)
        Cxq_alpha = -MAV.C_D_q*np.cos(self._alpha) + MAV.C_L_q*np.sin(self._alpha)
        Cxdele_alpha = -MAV.C_D_delta_e*np.cos(self._alpha) + MAV.C_L_delta_e*np.sin(self._alpha)
        Cz_alpha = -Cd_alpha*np.sin(self._alpha) - Cl_alpha*np.cos(self._alpha)
        Czq_alpha = -MAV.C_D_q*np.sin(self._alpha) - MAV.C_L_q*np.cos(self._alpha)
        Czdele_alpha = -MAV.C_D_delta_e*np.sin(self._alpha) - MAV.C_L_delta_e*np.cos(self._alpha)
        rhoVaS = 0.5*MAV.rho*(self._Va**2)*MAV.S_wing
        Fax = rhoVaS * (Cx_alpha + Cxq_alpha*MAV.c/(2*self._Va)*q + Cxdele_alpha*delta_e)
        Fay = rhoVaS * (MAV.C_Y_0 + MAV.C_Y_beta*self._beta + MAV.C_Y_p*MAV.b/(2*self._Va)*p \
            + MAV.C_Y_r*MAV.b/(2*self._Va)*r + MAV.C_Y_delta_a*delta_a + MAV.C_Y_delta_r*delta_r)
        Faz = rhoVaS * (Cz_alpha + Czq_alpha*MAV.c/(2*self._Va)*q + Czdele_alpha*delta_e)

        #forces from props
        Vin = MAV.V_max*delta_t
        a = MAV.rho*(MAV.D_prop**5)*MAV.C_Q0/(2*np.pi)**2
        b = MAV.rho*(MAV.D_prop**4)*MAV.C_Q1*self._Va/(2*np.pi) + MAV.KQ*MAV.K_V/MAV.R_motor
        c = MAV.rho*(MAV.D_prop**3)*MAV.C_Q2*self._Va**2 - MAV.KQ*Vin/MAV.R_motor + MAV.KQ*MAV.i0
        omega_p = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        Tp = (MAV.rho*(MAV.D_prop**4)*MAV.C_T0)*(omega_p**2) / (4*np.pi**2) + \
            (MAV.rho*(MAV.D_prop**3)*MAV.C_T1*self._Va*omega_p)/(2*np.pi) +\
            (MAV.rho*(MAV.D_prop**2)*MAV.C_T2*self._Va**2)

        #Total Forces
        fx = Fgx + Fax + Tp
        fy = Fgy + Fay
        fz = Fgz + Faz

        #moment from air
        Ma_x = rhoVaS * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta*self._beta + MAV.C_ell_p*MAV.b*p/(2*self._Va) + \
            MAV.C_ell_r*MAV.b*r/(2*self._Va) + MAV.C_ell_delta_a*delta_a + MAV.C_ell_delta_r*delta_r)
        Ma_y = rhoVaS * MAV.c * (MAV.C_m_0 + MAV.C_m_alpha*self._alpha + MAV.C_m_q*MAV.c*q/(2*self._Va) + \
            MAV.C_m_delta_e*delta_e)
        Ma_z = rhoVaS * MAV.b * (MAV.C_n_0 + MAV.C_n_beta*self._beta + MAV.C_n_p*MAV.b*p/(2*self._Va) + MAV.C_n_r*MAV.b*r/(2*self._Va) + \
            MAV.C_n_delta_a*delta_a + MAV.C_n_delta_r*delta_r)

        #moment from props
        Qp = (MAV.rho*(MAV.D_prop**5)*MAV.C_Q0*omega_p**2)/(4*np.pi**2) + \
            (MAV.rho*(MAV.D_prop**4)*MAV.C_Q1*self._Va*omega_p) / (2*np.pi) + \
            (MAV.rho*(MAV.D_prop**3)*MAV.C_Q2*self._Va**2)

        #Total Moment
        Mx = Ma_x + Qp
        My = Ma_y 
        Mz = Ma_z 
        #print(self._forces[0][0])
        self._forces[0][0] = fx
        self._forces[1][0] = fy
        self._forces[2][0] = fz
        fx = 0
        fy = 0
        #fz = 0
        Mx = 0
        My = 0
        Mz = 0

        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = self.phi
        self.msg_true_state.theta = self.theta
        self.msg_true_state.psi = self.psi
        self.msg_true_state.Vg = self.Vg #inertial velocity
        self.msg_true_state.gamma = self.gamma #flight path angle
        self.msg_true_state.chi = self.chi #course angle
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)
