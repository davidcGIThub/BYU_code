"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
        2/27/2020 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENS
from tools.tools import Euler2RotationMatrix
from tools.wrap import wrap
import parameters.aerosonde_parameters as MAV

from message_types.msg_state import msg_state

########################  Observer ######################## 

class observer:
    def __init__(self, ts_control):
        # initialized estimated state message
        self.estimated_state = msg_state()
        self.estimated_state.pn = MAV.pn0      # inertial north position in meters
        self.estimated_state.pe = MAV.pe0      # inertial east position in meters
        self.estimated_state.h = -MAV.pd0       # inertial altitude in meters
        self.estimated_state.phi = MAV.phi0    # roll angle in radians
        self.estimated_state.theta = MAV.theta0   # pitch angle in radians
        self.estimated_state.psi = MAV.psi0     # yaw angle in radians
        self.estimated_state.Va = MAV.Va0      # airspeed in meters/sec
        self.estimated_state.alpha = 0.   # angle of attack in radians
        self.estimated_state.beta = 0.    # sideslip angle in radians
        self.estimated_state.p = 0.       # roll rate in radians/sec
        self.estimated_state.q = 0.       # pitch rate in radians/sec
        self.estimated_state.r = 0.       # yaw rate in radians/sec
        self.estimated_state.Vg = 0.      # groundspeed in meters/sec
        self.estimated_state.gamma = 0.   # flight path angle in radians
        self.estimated_state.chi = 0.     # course angle in radians
        self.estimated_state.wn = 0.      # inertial windspeed in north direction in meters/sec
        self.estimated_state.we = 0.      # inertial windspeed in east direction in meters/sec
        self.estimated_state.bx = 0.      # gyro bias along roll axis in radians/sec
        self.estimated_state.by = 0.      # gyro bias along pitch axis in radians/sec
        self.estimated_state.bz = 0.      # gyro bias along yaw axis in radians/sec
        # use alpha filters to low pass filter gyros and accels
        self.lpf_gyro_x = alpha_filter(alpha=SENS.gyro_alpha)
        self.lpf_gyro_y = alpha_filter(alpha=SENS.gyro_alpha)
        self.lpf_gyro_z = alpha_filter(alpha=SENS.gyro_alpha)
        self.lpf_accel_x = alpha_filter(alpha = SENS.accel_alpha)
        self.lpf_accel_y = alpha_filter(alpha = SENS.accel_alpha)
        self.lpf_accel_z = alpha_filter(alpha = SENS.accel_alpha)
        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_abs = alpha_filter(alpha=SENS.static_pres_alpha,y0=self.estimated_state.h*MAV.rho*MAV.gravity)
        self.lpf_diff = alpha_filter(alpha=SENS.diff_press_alpha,y0=MAV.rho*(self.estimated_state.Va**2)/2)
        # ekf for phi and theta
        self.attitude_ekf = ekf_attitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ekf_position()



    def update(self, measurements):

        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurements.gyro_x)
        self.estimated_state.q = self.lpf_gyro_y.update(measurements.gyro_y)
        self.estimated_state.r = self.lpf_gyro_z.update(measurements.gyro_z)

        #estimates for u, v, w  are low pass filter
        self.estimated_state.u = self.lpf_accel_x.update(measurements.accel_x)
        self.estimated_state.v = self.lpf_accel_y.update(measurements.accel_y)
        self.estimated_state.w = self.lpf_accel_y.update(measurements.accel_z)

        # invert sensor model to get altitude and airspeed
        self.estimated_state.h = self.lpf_abs.update(measurements.static_pressure)/(MAV.rho * MAV.gravity)
        self.estimated_state.Va = np.sqrt( 2*self.lpf_diff.update(measurements.diff_pressure)/MAV.rho )

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(self.estimated_state, measurements)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(self.estimated_state, measurements)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state

######################## Low Pass Filter ######################## 

class alpha_filter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    #if  u is noisy, then alpha should be close to one (don't trust measurement)
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.alpha * self.y + (1 - self.alpha) * u
        return self.y

######################## Attitude EKF ######################## 

class ekf_attitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self,x0 = np.zeros([2,1]), Q_tune = 0):
        self.Q_tune = Q_tune*np.identity(2)
        self.Q_u = (SENS.gyro_sigma**2) * np.identity(4) # input noise covariance matrix
        self.Q_u[3,3] = SENS.diff_pres_sigma**2
        self.R = (SENS.accel_sigma**2) * np.identity(3) # measurement noise covariance matrix
        self.N = 5 # number of prediction step per sample 
        self.xhat = x0 # initial state: phi, theta
        self.P = np.identity(2)
        self.Ts = SIM.ts_control/self.N

    def update(self, state, measurement): #estimated state , current measurement
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        phi = x[0,0]
        theta = x[1,0]
        _f = np.array([ [state.p + state.q*np.sin(phi)*np.tan(theta) + state.r*np.cos(phi)*np.tan(theta)],
                        [state.q*np.cos(phi) - state.r*np.sin(phi)] ])
        return _f

    def h(self, x, state):
        # measurement model y
        phi = x[0,0]
        theta = x[1,0]
        _h = np.array([ [state.q*state.Va*np.sin(theta) + MAV.gravity*np.sin(theta)],
                        [state.r*state.Va*np.cos(theta) - state.p*state.Va*np.sin(theta) - MAV.gravity*np.cos(theta)*np.sin(phi)],
                        [-state.q*state.Va*np.cos(theta) - MAV.gravity*np.cos(theta)*np.cos(phi)] ])
        return _h

    def propagate_model(self, state):

        # model propagation
        for i in range(0, self.N):
             # propagate model
            self.xhat = self.xhat + self.Ts*self.f(self.xhat,state)
            phi = self.xhat[0,0]
            theta = self.xhat[1,0]
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # compute G matrix for gyro noise
            G = np.array([[1 , np.sin(phi)*np.tan(theta) , np.cos(phi)*np.tan(theta) , 0],
                        [0 , np.cos(phi) , -np.sin(phi) , 0]])
            #Compute process covariance matrix
            Q = np.dot(np.dot(G,self.Q_u),G.T) + self.Q_tune
            # convert to discrete time models
            A_d = np.identity(2) + A*self.Ts + np.dot(A,A)*self.Ts**2
            Q_d = Q*self.Ts**2
            # update P with discrete time model
            self.P = np.dot(np.dot(A_d,self.P),A_d.T) + Q_d

    def measurement_update(self, state, measurement):
        # measurement updates
        threshold = 2.0
        h = self.h(self.xhat, state)
        C = jacobian(self.h, self.xhat, state)
        y = np.array([measurement.accel_x, measurement.accel_y, measurement.accel_z])
        for i in range(0, 3):
            if np.abs(y[i]-h[i,0]) < threshold:
                Ci = C[i][None,:]
                Li = np.dot(self.P,Ci.T) * 1/(self.R[i,i] + np.dot(np.dot(Ci,self.P) , Ci.T))
                temp = np.identity(2) - np.dot(Li,Ci)
                self.P = np.dot(np.dot(temp , self.P)  ,  (temp).T) + np.dot(Li,self.R[i,i]*Li.T) 
                self.xhat = self.xhat + np.dot(Li , (y[i] - h[i,0]))

######################## POSITION EKF ################################
class ekf_position:
    # implement continous-discrete EKF to estimate pn, pe, chi, Vg
    def __init__(self,x0 = np.array([MAV.pn0,MAV.pe0,MAV.Va0,0.0,0.0,0.0,MAV.psi0])[:,None],Q_tune = .5):
        self.Q = Q_tune*np.identity(7)
        self.R = np.zeros([6,6])
        np.fill_diagonal(self.R,[SENS.gps_n_sigma**2, SENS.gps_e_sigma**2, 
                                SENS.gps_Vg_sigma**2, SENS.gps_course_sigma_ave**2,
                                SENS.wind_sigma, SENS.wind_sigma])
        self.N = 20  # number of prediction steps per sample
        self.Ts = (SIM.ts_control / self.N)
        self.xhat = x0
        self.P = np.identity(7)
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999


    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.pn = self.xhat.item(0)
        state.pe = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        pn = x[0,0]
        pe = x[1,0]
        Vg = x[2,0]
        chi = x[3,0]
        wn = x[4,0]
        we = x[5,0]
        psi = x[6,0]
        chi_dot = state.q*np.sin(state.phi)/np.cos(state.theta) + state.r*np.cos(state.phi)/np.cos(state.theta)
        _f = np.array([[Vg*np.cos(chi)],
                        [Vg*np.sin(chi)],
                        [state.Va*chi_dot*(we*np.cos(state.psi) - wn*np.sin(state.psi))/Vg],
                        [(MAV.gravity/Vg)*np.tan(state.phi)*np.cos(chi-psi)],
                        [0],
                        [0],
                        [chi_dot]])
        return _f

    def h(self, x, state):
        # measurement model for gps measurements
        pn = x[0,0]
        pe = x[1,0]
        Vg = x[2,0]
        chi = x[3,0]
        wn = x[4,0]
        we = x[5,0]
        psi = x[6,0]
        _h = np.array([[pn],
                        [pe],
                        [Vg],
                        [chi],
                        [state.Va*np.cos(psi) + wn - Vg*np.cos(chi)],
                        [state.Va*np.sin(psi) + we - Vg*np.sin(chi)]])
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
             # propagate model
            self.xhat = self.xhat + self.Ts*self.f(self.xhat,state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # convert to discrete time models
            A_d = np.identity(7) + A*self.Ts + np.dot(A,A)*self.Ts**2
            Q_d = self.Q*self.Ts**2
            # update P with discrete time model
            self.P = np.dot(np.dot(A_d,self.P),A_d.T) + Q_d

    def measurement_update(self, state, measurement):
        # always update based on wind triangle pseudo measurement
        h = self.h(self.xhat, state)
        C = jacobian(self.h, self.xhat, state)
        y = np.array([measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course, 0, 0])
        for i in range(4, 6):
            Ci = C[i][None,:]
            Li = np.dot(self.P,Ci.T) * 1/(self.R[i,i] + np.dot(np.dot(Ci,self.P) , Ci.T))
            temp = np.identity(7) - np.dot(Li,Ci)
            self.P = np.dot(np.dot(temp , self.P)  ,  (temp).T) + np.dot(Li,self.R[i,i]*Li.T) 
            self.xhat = self.xhat + np.dot(Li , (y[i] - h[i,0]))

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):
            h = self.h(self.xhat, state)
            C = jacobian(self.h, self.xhat, state)
            y = np.array([measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course])
            for i in range(0, 4):
                Ci = C[i][None,:]
                Li = np.dot(self.P,Ci.T) * 1/(self.R[i,i] + np.dot(np.dot(Ci,self.P) , Ci.T))
                temp = np.identity(7) - np.dot(Li,Ci)
                if i == 4:
                    y[i] = wrap(y[i], h[i,0])
                self.P = np.dot(np.dot(temp , self.P)  ,  (temp).T) + np.dot(Li,self.R[i,i]*Li.T) 
                self.xhat = self.xhat + np.dot(Li , (y[i] - h[i,0]))
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

######################## Jacobian Function ######################## 

def jacobian(fun, x, state):
    # compute jacobian of fun with respect to x
    f = fun(x, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.01  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J