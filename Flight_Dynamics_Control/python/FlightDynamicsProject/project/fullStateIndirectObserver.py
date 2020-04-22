import sys 
import numpy as np 
sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENS
from tools.tools import Euler2RotationMatrix, AngularRate2AttitudeDeriv
from tools.wrap import wrap, wrapAngle
import parameters.aerosonde_parameters as MAV
from message_types.msg_state import msg_state
import pdb
from scipy.stats import chi2

########################  Observer ######################## 

class fullStateIndirectObserver:
    def __init__(self, ts_control):
        #states
        self.estimated_state = msg_state()
        self.estimated_state.pn = MAV.pn0_n      # inertial north position in meters
        self.estimated_state.pe = MAV.pe0_n      # inertial east position in meters
        self.estimated_state.h = -MAV.pd0_n       # inertial altitude in meters
        self.estimated_state.phi = MAV.phi0_n    # roll angle in radians
        self.estimated_state.theta = MAV.theta0_n   # pitch angle in radians
        self.estimated_state.psi = MAV.psi0_n     
        
        # yaw angle in radians
        self.estimated_state.Va = MAV.Va0_n      # airspeed in meters/sec
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

        # estimators
        self.directEKF = indirectExtendedKalmanFilter()
        self.lpf_gyro_x = alpha_filter(alpha=SENS.gyro_alpha)
        self.lpf_gyro_y = alpha_filter(alpha=SENS.gyro_alpha)
        self.lpf_gyro_z = alpha_filter(alpha=SENS.gyro_alpha)
        self.lpf_accel_x = alpha_filter(alpha=SENS.accel_alpha)
        self.lpf_accel_y = alpha_filter(alpha=SENS.accel_alpha)
        self.lpf_accel_z = alpha_filter(alpha=SENS.accel_alpha)

    def update(self, measurements):
        self.estimated_state.p = self.lpf_gyro_x.update(measurements.gyro_x)
        self.estimated_state.q = self.lpf_gyro_y.update(measurements.gyro_y)
        self.estimated_state.r = self.lpf_gyro_z.update(measurements.gyro_z)
        xhat = self.directEKF.update(self.estimated_state, measurements)
        return self.estimated_state

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

class indirectExtendedKalmanFilter:
    def __init__(self, N = 10, x0_err = np.zeros((1,14)).T, x0 = np.array([[MAV.pn0_n, MAV.pe0_n, MAV.pd0_n, MAV.u0_n, MAV.v0_n, MAV.w0_n,
                                     MAV.phi0_n, MAV.theta0_n, MAV.psi0_n, 0, 0, 0,0,0]]).T ,
                    std_pn = (SENS.gps_n_sigma)**2,
                    std_pe = (2*SENS.gps_e_sigma)**2,
                    std_pd =  (5*(SENS.static_pres_sigma/(MAV.rho*MAV.gravity)))**2,
                    std_u = (2*SENS.accel_sigma)**2,
                    std_v = (2*SENS.accel_sigma)**2,
                    std_w =  (2*SENS.accel_sigma)**2,
                    std_phi = np.radians(10)**2,
                    std_theta = np.radians(10)**2,
                    std_psi = np.radians(10)**2,
                    std_bx = np.radians(0.05)**2,
                    std_by = np.radians(0.05)**2,
                    std_bz = np.radians(0.05)**2,
                    std_wn = SENS.wind_sigma**2,
                    std_we = SENS.wind_sigma**2):
        self.N = N # number of prediction step per sample
        self.Ts = SIM.ts_control/self.N
        self.xhat = x0 #[pn, pe, pd, u ,v , w, phi, theta, psi, bx, by, bz, wn, we]
        self.xhat_err = x0_err
        self.P = np.identity(14)
        self.Q_tune = np.identity(14) #tunable process noise
        Q_tune_diag = [std_pn,std_pe,std_u,std_v,std_w,std_phi,std_theta,std_psi,std_bx,std_by,std_bz,std_wn,std_we]*0
        np.fill_diagonal(self.Q_tune,Q_tune_diag)
        self.Q_gyro = np.identity(3) 
        np.fill_diagonal(self.Q_gyro, [SENS.gyro_sigma**2, SENS.gyro_sigma**2, SENS.gyro_sigma**2])
        self.Q_accel = np.identity(3)
        np.fill_diagonal(self.Q_accel, [SENS.accel_sigma**2, SENS.accel_sigma**2, SENS.accel_sigma**2])
        self.R = np.identity(7) #measurement noise
        np.fill_diagonal(self.R,[(10*SENS.static_pres_sigma)**2, (15*SENS.diff_pres_sigma)**2, (15*SENS.psuedo_ur_sigma)**2, 
                                 (2*SENS.gps_n_sigma)**2, (2*SENS.gps_e_sigma)**2, (2*SENS.gps_Vg_sigma)**2, (20*SENS.gps_course_sigma_ave)**2])
        yave = np.zeros((7,1)) #average of new measurements, used to check for outliers
        self.firstStep = True
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999
    
    def update(self, est_state, measurements):
        #check if this is the first measurement
        if self.firstStep:
            self.yave = np.array([[measurements.static_pressure, measurements.diff_pressure, 0,
                    measurements.gps_n, measurements.gps_e, measurements.gps_Vg, measurements.gps_course]]).T
            self.firstStep = False
        #treat accel and gyro measurements as an input to the propagation
        u = np.array([[measurements.accel_x, measurements.accel_y, measurements.accel_z, 
                        measurements.gyro_x, measurements.gyro_y, measurements.gyro_z ]]).T
        self.propagate_model(u)
        self.measurement_update(measurements)
        est_state.pn = self.xhat.item(0)
        est_state.pe = self.xhat.item(1)
        est_state.h = -self.xhat.item(2)
        est_state.u = self.xhat.item(3)
        est_state.v = self.xhat.item(4)
        est_state.w = self.xhat.item(5)
        est_state.phi = self.xhat.item(6)
        est_state.theta = self.xhat.item(7)
        est_state.psi = self.xhat.item(8)
        est_state.bx = self.xhat.item(9)
        est_state.by = self.xhat.item(10)
        est_state.bz = self.xhat.item(11)
        est_state.wn = self.xhat.item(12)
        est_state.we = self.xhat.item(13)
        # calculate remaining state estimates
        V = np.array([[est_state.u,est_state.v,est_state.w]]).T
        W = np.array([[est_state.wn,est_state.we,0]]).T
        R = Euler2RotationMatrix(est_state.phi,est_state.theta,est_state.psi)
        Vg = np.dot(R,V)
        Va = V-np.dot(R.T,W)
        ur = Va.item(0)
        vr = Va.item(1)
        wr = Va.item(2)
        est_state.Vg = np.sqrt(Vg.item(0)**2 + Vg.item(1)**2)
        est_state.Va = np.linalg.norm(Va)
        est_state.alpha = np.arctan2(wr,ur)
        if est_state.Va == 0:
            est_state.beta = 0
        else:
            est_state.beta = np.arcsin(vr/(est_state.Va))
        est_state.gamma = np.arctan2(Vg.item(2),Vg.item(0))
        est_state.chi = np.arctan2(Vg.item(1),Vg.item(0))

    def propagate_model(self, u_):
        for i in range(0, self.N):
             # propagate model
            self.xhat = self.xhat + self.Ts*self.f(self.xhat, u_)
            #wrap angless
            # self.xhat[6] = wrapAngle(self.xhat[6])
            # self.xhat[7] = wrapAngle(self.xhat[7])
            # self.xhat[8] = wrapAngle(self.xhat[8])
            #prevent division by zero error
            if np.abs(np.cos(self.xhat[7])) < 0.0001:
                self.xhat[7] = 1.57 * np.sign(self.xhat[7])
            # compute Jacobian
            #A = self.jacobian(self.f,self.xhat,u_)
            A = self.f_jacobian(self.xhat, u_)
            #propogate error
            self.xhat_err = self.xhat_err + np.dot(A,self.xhat_err)
            # compute G matrix for gyro noise
            u = self.xhat.item(3)
            v = self.xhat.item(4)
            w = self.xhat.item(5)
            phi = self.xhat.item(6)
            theta = self.xhat.item(7)
            psi = self.xhat.item(8)
            #Jacobian of dxhat with respect to input (Gyroscope)
            Gg = np.array([[0 , 0 , 0],
                            [0 , 0 , 0],
                            [0 , 0 , 0],
                            [0 , -w , v],
                            [w , 0 , -u],
                            [-v , u , 0],
                            [1 , np.sin(phi)*np.tan(theta) , np.cos(phi)*np.tan(theta)],
                            [0 , np.cos(phi) , -np.sin(phi)],
                            [0 , np.sin(phi)/np.cos(theta) , np.cos(phi)/np.cos(theta)],
                            [0 , 0 , 0],
                            [0 , 0 , 0],
                            [0 , 0 , 0],
                            [0 , 0 , 0],
                            [0 , 0 , 0]])
            # compute G matrix for accel noise
            #Jacobian of dxhat with respect to input (accelerometers)
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
            #P_dot = np.dot(A,self.P) + np.dot(self.P,A.T) + Q
            #self.P = self.P + self.Ts*(P_dot)

    def measurement_update(self, measurement):
        # always update pressure measurements, and fake sideslip
        u_ = np.array([])
        h = self.h(self.xhat,u_)
        #C = self.jacobian(self.h, self.xhat, u_)
        C = self.h_jacobian(self.xhat) 
        psuedo_vr = 0 #drive vr to zero so that beta = 0
        y = np.array([[measurement.static_pressure, measurement.diff_pressure, psuedo_vr,
                    measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course]]).T
        #wrap chi to be within pi of chi measured
        y[6] = wrap(y[6], h[6,0])
        for i in range(0, 3):
            Ci = C[i][None,:]
            if self.checkOutlier(self.R[i,i], Ci, self.P ,y[i],h[i]) or self.checkOutlier(self.R[i,i], Ci, self.P ,y[i],self.yave[i]):
                Li = np.dot(self.P,Ci.T) * 1/(self.R[i,i] + np.dot(np.dot(Ci,self.P) , Ci.T))
                temp = np.identity(14) - np.dot(Li,Ci)
                self.P = np.dot(np.dot(temp , self.P)  ,  (temp).T) + np.dot(Li,self.R[i,i]*Li.T) 
                self.xhat_err = self.xhat_err + np.dot(Li , (y.item(i) - h.item(i) - np.dot(Ci,self.xhat_err))) 


        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
             or (measurement.gps_e != self.gps_e_old) \
             or (measurement.gps_Vg != self.gps_Vg_old) \
             or (measurement.gps_course != self.gps_course_old):
            for i in range(3,7):
                Ci = C[i][None,:]
                if self.checkOutlier(self.R[i,i], Ci, self.P ,y[i],h[i]) or self.checkOutlier(self.R[i,i], Ci, self.P ,y[i],self.yave[i]):
                    Li = np.dot(self.P,Ci.T) * 1/(self.R[i,i] + np.dot(np.dot(Ci,self.P) , Ci.T))
                    temp = np.identity(14) - np.dot(Li,Ci)
                    self.P = np.dot(np.dot(temp , self.P)  ,  temp.T) + np.dot(Li,self.R[i,i]*Li.T) 
                    self.xhat_err = self.xhat_err + np.dot(Li , (y.item(i) - h.item(i) - np.dot(Ci,self.xhat_err)))
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course
        #update average of measurements
        self.yave = (self.yave + y)/2
        #update estimate based on error
        self.xhat = self.xhat + self.xhat_err
        self.xhat_err = np.zeros((14,1))
    
    def f(self, x, u_):
    # system dynamics for propagation model: xdot = f(x, u)
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
                    [(np.cos(theta)*np.sin(psi))*u + (np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi))*v \
                        + (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi))*w ],
                    [-u*np.sin(theta) + v*np.sin(phi)*np.cos(theta) + w*np.cos(phi)*np.cos(theta)],
                    [accel_x - MAV.gravity*np.sin(theta) + v*(gyro_z -bz) - w*(gyro_y - by)],
                    [accel_y + MAV.gravity*np.cos(theta)*np.sin(phi) + w*(gyro_x - bx) - u*(gyro_z - bz)],
                    [accel_z + MAV.gravity*np.cos(theta)*np.cos(phi) + u*(gyro_y -by) - v*(gyro_x - bx)],
                    [(gyro_x-bx) + np.sin(phi)*np.tan(theta)*(gyro_y-by) + np.cos(phi)*np.tan(theta)*(gyro_z-bz)],
                    [np.cos(phi)*(gyro_y-by) - np.sin(phi)*(gyro_z-bz)],
                    [np.sin(phi)*(gyro_y-by)/np.cos(theta) + np.cos(phi)*(gyro_z-bz)/np.cos(theta)],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0]])
        return _f

    def f_jacobian(self,x, u_):
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
        c_phi = np.cos(phi)
        c_theta = np.cos(theta)
        c_psi = np.cos(psi)
        s_phi = np.sin(phi)
        s_theta = np.sin(theta)
        s_psi = np.sin(psi)
        t_theta = np.tan(theta)
        V = np.array([[u,v,w]]).T
        R = Euler2RotationMatrix(phi,theta,psi)
        g = np.array([[0,0,MAV.gravity]]).T
        gyro_unbiased = np.array([[gyro_x,gyro_y,gyro_z]]).T - np.array([[bx,by,bz]]).T
        dR_dphi = np.array([[0 , c_phi*s_theta*c_psi + s_phi*s_psi , -s_phi*s_theta*c_psi+c_phi*s_psi],
                            [0 , c_phi*s_theta*s_psi-s_phi*c_psi , -s_phi*s_theta*s_psi-c_phi*c_psi],
                            [0 , c_phi*c_theta , -s_phi*c_theta]])
        dR_dtheta = np.array([[-s_theta*c_psi , s_phi*c_theta*c_psi , c_phi*c_theta*c_psi],
                            [-s_theta*s_psi , s_phi*c_theta*s_psi , c_phi*c_theta*s_psi],
                            [-c_theta , -s_phi*s_theta , -c_phi*s_theta]])
        dR_dpsi = np.array([[-c_theta*s_psi , -s_phi*s_theta*s_psi-c_phi*c_psi , -c_phi*s_theta*s_psi + s_phi*c_psi],
                            [c_theta*c_psi , s_phi*s_theta*c_psi-c_phi*s_psi , c_phi*s_theta*c_psi + s_phi*s_psi],
                            [0 , 0 , 0]])
        dS_dphi = np.array([[0 , c_phi*t_theta , -s_phi*t_theta],
                            [0 , -s_phi , -c_phi],
                            [0 , c_phi/c_theta , -s_phi/c_theta]])
        dS_dtheta = np.array([[0 , s_phi/c_theta**2 , c_phi/c_theta**2],
                            [0 , 0 , 0],
                            [0 , s_phi*t_theta/c_theta , c_phi*t_theta/c_theta]])
        dS_dpsi = np.zeros((3,3))
        #derivative of position rows
        d_RV_dAngles = np.concatenate((np.dot(dR_dphi , V) , np.dot(dR_dtheta , V) , np.dot(dR_dpsi, V)) , axis = 1)
        dP_dx = np.concatenate((np.zeros((3,3)) , R , d_RV_dAngles , np.zeros((3,3)) , np.zeros((3,2))) , axis=1)
        #derivative of velocity rows
        d_RTg_dAngles = np.concatenate((np.dot(dR_dphi.T , g) , np.dot(dR_dtheta.T , g) , np.dot(dR_dpsi.T, g)) , axis = 1)
        gyro_mat = self.arr_to_xmat(-gyro_unbiased)
        V_mat = self.arr_to_xmat(-V)
        dV_dx = np.concatenate( (np.zeros((3,3)) , gyro_mat , d_RTg_dAngles , V_mat , np.zeros((3,2)) ) , axis=1)
        #derivative of attitude rows
        S = AngularRate2AttitudeDeriv(phi,theta,psi)
        dSgyro_dAngles = np.concatenate((np.dot(dS_dphi,gyro_unbiased) , np.dot(dS_dtheta,gyro_unbiased) , np.dot(dS_dpsi,gyro_unbiased) ), axis=1)
        dAngles_dx = np.concatenate((np.zeros((3,3)) , np.zeros((3,3)), dSgyro_dAngles , -S , np.zeros((3,2))),axis=1)
        #derivative of gyro bias rows
        dGyroBias_dx = np.zeros((3,14))
        #derivative of wind rows
        dWind_dx = np.zeros((2,14))
        A = np.concatenate((dP_dx , dV_dx , dAngles_dx , dGyroBias_dx , dWind_dx ), axis=0)
        return A

    def h(self,x, u_):
        pn = x.item(0)
        pe = x.item(1)
        pd = x.item(2)
        u = x.item(3)
        v = x.item(4)
        w = x.item(5)
        phi = x.item(6)
        theta = x.item(7)
        psi = x.item(8)
        wn = x.item(12)
        we = x.item(13)
        V = np.array([[u,v,w]]).T
        W = np.array([[wn,we,0]]).T
        R = Euler2RotationMatrix(phi,theta,psi)
        Vg = np.dot(R,V)
        Vn = Vg.item(0) #north velocity
        Ve = Vg.item(1) #east velocity
        Va = V-np.dot(R.T,W)
        vr = Va.item(1)
        h_ = np.array([[-MAV.rho*MAV.gravity*pd],   #h_static
                    [.5*MAV.rho*np.dot(Va.T,Va).item(0)],   #h_diff
                    [vr],                           #h_beta - driving side dir Va zero
                    [pn],                           #h_gps_n 
                    [pe],                           #h_gps_e
                    [np.sqrt(Ve**2 + Vn**2)],    #h_gps_Vg
                    [np.arctan2(Ve,Vn)]])           #h_chi
        return h_

    def h_jacobian(self, x): 
        #Jacobian of dxhat with respect to the measurements
        #Create variables used for calculations
        pn = x.item(0)
        pe = x.item(1)
        pd = x.item(2)
        u = x.item(3)
        v = x.item(4)
        w = x.item(5)
        phi = x.item(6)
        theta = x.item(7)
        psi = x.item(8)
        wn = x.item(12)
        we = x.item(13)
        V = np.array([[u,v,w]]).T
        W = np.array([[wn,we,0]]).T
        R = Euler2RotationMatrix(phi,theta,psi)
        Vg = np.dot(R,V)
        Vn = Vg.item(0) #north velocity
        Ve = Vg.item(1) #east velocity
        Vg_horiz = np.array([[Vn,Ve, 0]]).T
        Va = V-np.dot(R.T,W)
        c_phi = np.cos(phi)
        c_theta = np.cos(theta)
        c_psi = np.cos(psi)
        s_phi = np.sin(phi)
        s_theta = np.sin(theta)
        s_psi = np.sin(psi)
        P_mat = np.array([[1,0,0],[0,1,0]])
        #Calculate Jacobian
        #Cstatic
        C_static = np.array([ 0 , 0 , -MAV.rho*MAV.gravity , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ])
        #Cdiff Calculations
        dRW_dAtt = np.array([[0 , -wn*s_theta*c_psi-we*s_theta*s_psi , -wn*c_theta*s_psi+we*c_theta*c_psi],
                                [wn*(c_phi*s_theta*c_psi+s_phi*s_psi) + we*(c_phi*s_theta*s_psi-s_phi*c_psi) , wn*s_phi*c_theta*c_psi + we*s_phi*c_theta*s_psi ,
                                            wn*(-s_phi*s_theta*s_psi-c_phi*c_psi) + we*(s_phi*s_theta*c_psi-c_phi*s_psi)],
                                [wn*(-s_phi*s_theta*c_psi+c_phi*s_psi) + we*(-s_phi*s_theta*s_psi-c_phi*c_psi) , wn*c_phi*c_theta*c_psi + we*c_phi*c_theta*s_psi , 
                                            wn*(-c_phi*s_theta*s_psi+s_phi*c_psi) + we*(c_phi*s_theta*c_psi+s_phi*s_psi)]])
        diff_Att = np.dot(-dRW_dAtt.T,Va)
        diff_Wind = -np.dot(P_mat , np.dot(R,Va))
        C_diff = MAV.rho*np.array([ 0 , 0 , 0 , Va.item(0), Va.item(1), Va.item(2), diff_Att.item(0), diff_Att.item(1), diff_Att.item(2), 0 , 0 , 0 , diff_Wind.item(0), diff_Wind.item(1)])
        #C_beta - sideslip pseudo
        dvr_dphi = -wn*(c_phi*s_theta*c_psi+s_phi*s_psi) - we*(c_phi*s_theta*s_psi-s_phi*c_psi)
        dvr_dtheta = -wn*s_phi*c_theta*c_psi - we*s_phi*c_theta*s_psi
        dvr_dpsi =  -wn*(-s_phi*s_theta*s_psi-c_phi*c_psi) - we*(s_phi*s_theta*c_psi-c_phi*s_psi)
        dvr_dwn = -(c_phi*s_theta*c_psi + s_phi*s_psi)
        dvr_dwe = (c_phi*s_theta*s_psi-s_phi*c_psi)
        C_beta = np.array([0 , 0 , 0 , 0 , 1 , 0 , dvr_dphi , dvr_dtheta , dvr_dpsi , 0 , 0 , 0 , dvr_dwn , dvr_dwe])
        #C_gps_n
        C_gps_n = np.array([ 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0])
        #C_gps_e
        C_gps_e = np.array([ 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0])
        #C_gps_Vg calculations
        gps_Vg_Vel = np.dot( R.T, np.dot(P_mat.T , np.dot(P_mat , np.dot(R,V)))) / np.linalg.norm(Vg_horiz)
        temp_numerator = 2*np.linalg.norm(Vg_horiz)
        dVn_dphi = v*(c_phi*s_theta*c_psi+s_phi*s_psi) + w*(-s_phi*s_theta*c_psi+c_phi*s_psi)
        dVn_dtheta = u*(-s_theta*c_psi) + v*(s_phi*c_theta*c_psi) + w*(c_phi*c_theta*c_psi)
        dVn_dpsi = u*(-c_theta*s_psi) + v*(-s_phi*s_theta*s_psi-c_phi*c_psi) + w*(-c_phi*s_theta*s_psi+s_phi*c_psi)
        dVe_dphi = v*(c_phi*s_theta*s_psi-s_phi*c_psi) + w*(-s_phi*s_theta*s_psi-c_phi*c_psi)
        dVe_dtheta = u*(-s_theta*s_psi) + v*(s_phi*c_theta*s_psi) + w*(c_phi*c_theta*s_psi)
        dVe_dpsi =  u*(c_theta*c_psi) + v*(s_phi*s_theta*c_psi-c_phi*s_psi) + w*(c_phi*s_theta*c_psi+s_phi*s_psi)
        dVgmag_dphi = (Vn*dVn_dphi + Ve*dVe_dphi)/(np.linalg.norm(Vg_horiz))
        dVgmag_dtheta = (Vn*dVn_dtheta + Ve*dVe_dtheta)/(np.linalg.norm(Vg_horiz))
        dVgmag_dpsi = (Vn*dVn_dpsi + Ve*dVe_dpsi)/(np.linalg.norm(Vg_horiz))
        C_gps_Vg = np.array([0 , 0 , 0 , gps_Vg_Vel.item(0), gps_Vg_Vel.item(1) , gps_Vg_Vel.item(2) , dVgmag_dphi , dVgmag_dtheta , dVgmag_dpsi , 0 , 0 , 0 , 0 , 0])
        #Calculations for C_gps_chi
        (Vn**2 + Ve**2)
        dVn_du = R[0,0]
        dVn_dv = R[0,1]
        dVn_dw = R[0,2]
        dVe_du = R[1,0]
        dVe_dv = R[1,1]
        dVe_dw = R[1,2]
        dchi_du = (Vn*dVe_du - Ve*dVn_du)/(Vn**2 + Ve**2)
        dchi_dv = (Vn*dVe_dv - Ve*dVn_dv)/(Vn**2 + Ve**2)
        dchi_dw = (Vn*dVe_dw - Ve*dVn_dw)/(Vn**2 + Ve**2)
        dchi_dphi = (Vn*dVe_dphi - Ve*dVn_dphi)/(Vn**2 + Ve**2)
        dchi_dtheta = (Vn*dVe_dtheta - Ve*dVn_dtheta)/(Vn**2 + Ve**2)
        dchi_dpsi = (Vn*dVe_dpsi - Ve*dVn_dpsi)/(Vn**2 + Ve**2)
        C_gps_chi = np.array([0 , 0 , 0 , dchi_du , dchi_dv , dchi_dw , dchi_dphi , dchi_dtheta , dchi_dpsi, 0 , 0 , 0 , 0 ,0])
        C = np.array([C_static , C_diff , C_beta , C_gps_n , C_gps_e , C_gps_Vg , C_gps_chi])
        dVg_du = (Vn*dVn_du + Ve*dVe_du)/np.linalg.norm(Vg_horiz)
        return C

    def checkOutlier(self, R, C, P,y,h, prob=0.01, df = 1):
        S_inv = np.linalg.inv(R + np.dot(np.dot(C,P),C.T))
        if chi2.sf( np.dot((y-h).T , np.dot(S_inv,(y-h))) , df ) > prob:
            return True
        else:
            return False

    def arr_to_xmat(self,arr):
        arr1 = arr.item(0)
        arr2 = arr.item(1)
        arr3 = arr.item(2)
        xmat = np.array([[0, -arr3 , arr2],
                        [arr3 , 0 , -arr1],
                        [-arr2 , arr1 , 0]])
        return xmat

    def jacobian(self, fun, x, u_):
        # compute jacobian of fun with respect to x
        f = fun(x, u_)
        m = f.shape[0]
        n = x.shape[0]
        eps = 0.0001  # deviation
        J = np.zeros((m, n))
        #looping through each x estimate to take derivative of H with respect to x
        for i in range(0, n):
            x_eps = np.copy(x)
            x_eps[i][0] += eps
            f_eps = fun(x_eps, u_)
            df = (f_eps - f) / eps
            J[:, i] = df[:, 0]
        return J


# dVn_dphi = v*(c_phi*s_theta*c_psi+s_phi*s_psi) + w*(-s_phi*s_theta*c_psi+c_phi*s_psi)
# dVn_dtheta = u*(-s_theta*c_psi) + v*(s_phi*c_theta*c_psi) + w*(c_phi*c_theta*c_psi)
# dVn_dpsi = u*(-c_theta*s_psi) + v*(-s_phi*s_theta*s_psi-c_phi*c_psi) + w*(-c_phi*s_theta*s_psi+s_phi*c_psi)
# dVe_dphi = v*(c_phi*s_theta*s_psi-s_phi*c_psi) + w*(-s_phi*s_theta*s_psi-c_phi*c_psi)
# dVe_dtheta = u*(-s_theta*s_psi) + v*(s_phi*c_theta*s_psi) + w*(c_phi*c_theta*s_psi)
# dVe_dpsi =  u*(c_theta*c_psi) + v*(s_phi*s_theta*c_psi-c_phi*s_psi) + w*(c_phi*s_theta*c_psi+s_phi*s_psi)
# dVd_dphi = v*(c_phi*c_theta) + w*(-s_phi*c_theta)
# dVd_dtheta = u*(-c_theta) + v*(-s_phi*s_theta) + w*(-c_phi*s_theta)
# dVd_dpsi = 0
