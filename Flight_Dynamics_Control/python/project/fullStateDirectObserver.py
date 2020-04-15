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

class fullStateDirectObserver:
    def __init__(self, ts_control):
        self.estimated_state = msg_state()
        self.estimated_state.pn = 0.      # inertial north position in meters
        self.estimated_state.pe = 0.      # inertial east position in meters
        self.estimated_state.h = 0.       # inertial altitude in meters
        self.estimated_state.phi = 0.     # roll angle in radians
        self.estimated_state.theta = 0.   # pitch angle in radians
        self.estimated_state.psi = 0.     # yaw angle in radians
        self.estimated_state.Va = 0.      # airspeed in meters/sec
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
        self.directEKF = directExtendedKalmanFilter()

    def update(self, measurements):
        self.estimated_state.p = self.lpf_gyro_x.update(measurements.gyro_x)
        self.estimated_state.q = self.lpf_gyro_y.update(measurements.gyro_y)
        self.estimated_state.r = self.lpf_gyro_z.update(measurements.gyro_z)
        self.directEKF.update(self.estimated_state, measurements)
        # self.gyro_x = 0  # gyroscope along body x axis
        # self.gyro_y = 0  # gyroscope along body y axis
        # self.gyro_z = 0  # gyroscope along body z axis
        # self.accel_x = 0  # specific acceleration along body x axis
        # self.accel_y = 0  # specific acceleration along body y axis
        # self.accel_z = 0  # specific acceleration along body z axis
        # self.mag_x = 0  # magnetic field along body x axis
        # self.mag_y = 0  # magnetic field along body y axis
        # self.mag_z = 0  # magnetic field along body z axis
        # self.static_pressure = 0  # static pressure
        # self.diff_pressure = 0  # differential pressure
        # self.gps_n = 0  # gps north
        # self.gps_e = 0  # gps east
        # self.gps_h = 0  # gps altitude
        # self.gps_Vg = 0  # gps ground speed
        # self.gps_course = 0  # gps course angle

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

class directExtendedKalmanFilter:
    def __init__(self, x0 = np.array([[MAV.pn0, MAV.pe0, MAV.pd, MAV.Va0, MAV.v0, MAV.w0,
                                     MAV.phi0, MAV.theta0, MAV.psi0, 0, 0, 0,0,0]]).T ,
                    std_pn = SENS.gps_n_sigma**2,
                    std_pe = SENS.gps_e_sigma**2,
                    std_pd =  ((SENS.gps_h_sigma+(MAV.rho*MAV.gravity*SENS.static_pres_sigma))/2.0)**2,
                    std_u = (SENS.gps_Vg_sigma**2)/3,
                    std_v = (SENS.gps_Vg_sigma**2)/3,
                    std_w =  (SENS.gps_Vg_sigma**2)/3,
                    std_phi = np.radians(0.1)**2,
                    std_theta = np.radians(0.1)**2,
                    std_psi = SENS.mag_sigma**2, 
                    std_bx = np.radians(0.05)**2, 
                    std_by = np.radians(0.05)**2,
                    std_bz = np.radians(0.05)**2,
                    std_wn = SENS.sigma_wind**2,
                    std_we = SENS.sigma_wind**2):
        self.N = 5 # number of prediction step per sample
        self.Ts = SIM.ts_control/self.N
        self.xhat = x0 #[pn, pe, pd, u ,v , w, phi, theta, psi, bx, by, bz, wn, we]
        self.P = np.identity(14)
        self.Q_tune = np.identity()
        np.fill_diagonal(self.Q_tune,[std_pn,std_pe,std_u,std_v,std_w,std_phi,std_theta,std_psi,
                                        std_bx,std_by,std_bz,std_wn,std_we])
        self.Q_gyro = np.identity(3)
        np.fill_diagonal(self.Q_gyro, [SENS.gyro_sigma**2, SENS.gyro_sigma**2, SENS.gyro_sigma**2])
        self.Q_accel = np.identity(3)
        np.fill_diagonal(self.Q_accel, [SENS.gyro_accel**2, SENS.gyro_accel**2, SENS.gyro_accel**2])
        self.R = np.identity(7)
        np.fill_diagonal(self.R,[SENS.static_pres_sigma, SENS.diff_pres_sigma, SENS.beta_sigma, 
                                 SENS.gps_n_sigma, SENS.gps_e_sigma, SENS.gps_Vg_sigma, SENS.gps_course_sigma_ave])
    
    def update(self, est_state, measurements):
        #treat accel and gyro measurements as an input to the propagation
        u = np.array([[measurements.accel_x, measurements.accel_y, measurements.accel_z, 
                        measurements.gyro_x, measurements.gyro_y, measurements.gyro_z ]]).T
        self.propagate_model(u)
        self.measurement_update(measurements)
        est_state.pn = self.xhat.item(0)
        est_state.pe = self.xhat.item(1)
        est_state.h = self.xhat.item(2)
        est_state.u = self.xhat.item(3)
        est_state.v = self.xhat.item(4)
        est_state.w = self.xhat_item(5)
        est_state.phi = self.xhat.item(6)
        est_state.theta = self.xhat.item(7)
        est_state.psi = self.xhat.item(8)
        est_state.bx = self.xhat.item(9)
        est_state.by = self.xhat.item(10)
        est_state.bz = self.xhat.item(11)
        est_state.wn = self.xhat.item(12)
        est_state.we = self.xhat.item(13)

    def propagate_model(self, u_):
        for i in range(0, self.N):
             # propagate model
            self.xhat = self.xhat + self.Ts*self.f(self.xhat, u_)
            # compute Jacobian
            A = jacobian(self.f,self.xhat,u_)
            # compute G matrix for gyro noise
            u = self.xhat.item(3)
            v = self.xhat.item(4)
            w = self.xhat_item(5)
            phi = self.xhat.item(6)
            theta = self.xhat.item(7)
            psi = self.xhat.item(8)
            Gg = np.array([[0 , 0 , 0],
                            [0 , 0 , 0],
                            [0 , 0 , 0],
                            [0 , -w , v],
                            [w , 0 , -u],
                            [-v , u , 0],
                            [1 , np.sin(phi)*np.tan(theta) , np.cos(phi)*np.tan(theta)],
                            [0 , np.cos(phi) , -np.sin(phi)],
                            [0 , np.sin(phi)*np.sec(theta) , np.cos(phi)*np.sec(theta)],
                            [0 , 0 , 0],
                            [0 , 0 , 0],
                            [0 , 0 , 0],
                            [0 , 0 , 0],
                            [0 , 0 , 0]])
            # compute G matrix for accel noise
            Ga = np.zeros([14,3])
            Ga[3,0] = -1
            Ga[4,1] = -1
            Ga[5,2] = -1
            #Compute process covariance matrix
            Q = np.dot(np.dot(Gg,self.Q_gyro),Gg.T) + np.dot(np.dot(Ga,self.Q_accel),Ga.T) + self.Q_tune
            #convert to discrete time models
            A_d = np.identity(14) + A*self.Ts + np.dot(A,A)*self.Ts**2
            Q_d = Q*self.Ts**2
            # update P with discrete time model
            self.P = np.dot(np.dot(A_d,self.P),A_d.T) + Q_d

    def f(self, x, u_):
    # system dynamics for propagation model: xdot = f(x, u)
        #y[0] = accel_x
        pn = x.item(0)
        pe = x.item(1)
        pd = x.item(2)
        u = x.item(3)
        v = x.item(4)
        w = x.item(5)
        phi = x.item(6)
        theta = x.item(7)
        psi = x.item(8)
        bx = x.item(9)
        by = x.item(10)
        bz = x.item(11)
        wn = x.item(12)
        we = x.item(13)
        accel_x = u_.item(0)
        accel_y = u_.item(1)
        accel_z = u_.item(2)
        gyro_x = u_.item(3)
        gyro_y = u_.item(4)
        gyro_z = u_.item(5)
        _f = np.array([[np.cos(theta)*np.cos(psi)*u + (np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi))*v \
                         + (np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi))*w],
                    [(np.cos(theta)*np.sin(psi))*u - (np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi))*v \
                        + (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi))*w ],
                    [-u*np.sin(theta) - v*np.sin(phi)*np.cos(theta) - w*np.cos(phi)*np.cos(theta)],
                    [-w*(gyro_y - by) + v*(gyro_z -bz) + accel_x - MAV.g*np.sin(theta)],
                    [w*(gyro_x - bx) - u*(gyro_z - bz) + accel_y - MAV.g*np.cos(theta)*np.sin(phi)],
                    [-v*(gyro_x - bx) + u*(gyro_y -by) + accel_z - MAV.g*np.cos(theta)*np.cos(phi)],
                    [(gyro_x-bx) + np.sin(phi)*np.tan(theta)*(gyro_y-by) + np.cos(phi)*np.tan(theta)*(gyro_z-bz)],
                    [np.cos(phi)*(gyro_y-by) - np.sin(phi)*(gyro_z-bz)],
                    [np.sin(phi)*np.sec(theta)*(gyro_y-by) + np.cos(phi)*np.sec(theta)*(gyro_z-bz)],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0]])
        return _f

    def measurement_update(self, state, measurement):
        # always update based on wind triangle pseudo measurement
        # h = self.h(self.xhat, state)
        # C = jacobian(self.h, self.xhat, state)
        # y = np.array([measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course, 0, 0])
        # for i in range(4, 6):
        #     Ci = C[i][None,:]
        #     Li = np.dot(self.P,Ci.T) * 1/(self.R[i,i] + np.dot(np.dot(Ci,self.P) , Ci.T))
        #     temp = np.identity(7) - np.dot(Li,Ci)
        #     self.P = np.dot(np.dot(temp , self.P)  ,  (temp).T) + np.dot(Li,self.R[i,i]*Li.T) 
        #     self.xhat = self.xhat + np.dot(Li , (y[i] - h[i,0]))

        # # only update GPS when one of the signals changes
        # if (measurement.gps_n != self.gps_n_old) \
        #     or (measurement.gps_e != self.gps_e_old) \
        #     or (measurement.gps_Vg != self.gps_Vg_old) \
        #     or (measurement.gps_course != self.gps_course_old):

    def h(self,):

    def jacobian(fun, x, u_):
        # compute jacobian of fun with respect to x
        f = fun(x, u_)
        m = f.shape[0]
        n = x.shape[0]
        eps = 0.01  # deviation
        J = np.zeros((m, n))
        for i in range(0, n):
            x_eps = np.copy(x)
            x_eps[i][0] += eps
            f_eps = fun(x_eps, u_)
            df = (f_eps - f) / eps
            J[:, i] = df[:, 0]
        return J



