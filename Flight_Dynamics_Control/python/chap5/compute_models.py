"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion, Quaternion2Euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from chap4.mav_dynamics import mav_dynamics
import parameters.simulation_parameters as SIM


def compute_model(trim_state, trim_input):
    # A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, beta_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.item(0), trim_input.item(1), trim_input.item(2), trim_input.item(3)))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('beta_trim = %f\n' % beta_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    # file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f]])\n' %
    # (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
    #  A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
    #  A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
    #  A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
    #  A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    # file.write('B_lon = np.array([\n    [%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f]])\n' %
    # (B_lon[0][0], B_lon[0][1],
    #  B_lon[1][0], B_lon[1][1],
    #  B_lon[2][0], B_lon[2][1],
    #  B_lon[3][0], B_lon[3][1],
    #  B_lon[4][0], B_lon[4][1],))
    # file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f]])\n' %
    # (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
    #  A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
    #  A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
    #  A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
    #  A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    # file.write('B_lat = np.array([\n    [%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f]])\n' %
    # (B_lat[0][0], B_lat[0][1],
    #  B_lat[1][0], B_lat[1][1],
    #  B_lat[2][0], B_lat[2][1],
    #  B_lat[3][0], B_lat[3][1],
    #  B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(trim_state, trim_input): #computes the transfer function linearized at trim
    # trim values
    mav = mav_dynamics(SIM.ts_simulation)
    mav.set_state(trim_state)
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    beta_trim = mav._beta
    phi, theta_trim, psi = Quaternion2Euler(trim_state[6:10])
    delta_t_trim = trim_input[3,0]
    _dT_dVa = dT_dVa(Va_trim, delta_t_trim)
    _dT_ddelta_t = dT_ddelta_t(Va_trim, delta_t_trim)

    # define transfer function constants
    a_phi1 = -0.25 * MAV.rho * mav._Va * MAV.S_wing * (MAV.b**2) * MAV.C_p_p
    a_phi2 = .5 * MAV.rho * (mav._Va**2) * MAV.S_wing * MAV.b * MAV.C_p_delta_a
    a_theta1 = - MAV.rho * mav._Va * (MAV.c**2) * MAV.S_wing * MAV.C_m_q / (4 * MAV.Jy)
    a_theta2 = - MAV.rho * (mav._Va**2) * MAV.c * MAV.S_wing * MAV.C_m_alpha / (2 * MAV.Jy)
    a_theta3 = MAV.rho * (mav._Va**2) * MAV.c * MAV.S_wing * MAV.C_m_delta_e / (2 * MAV.Jy)
    a_V1 = (MAV.rho * Va_trim * MAV.S_wing / MAV.mass) * (MAV.C_D_0 + MAV.C_D_alpha * alpha_trim + MAV.C_D_delta_e) - _dT_dVa/MAV.mass
    a_V2 = _dT_ddelta_t / MAV.mass
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)
    return Va_trim, alpha_trim,beta_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3

# def compute_ss_model(mav, trim_state, trim_input):
#     x_euler = euler_state(trim_state)
#     return A_lon, B_lon, A_lat, B_lat

# def euler_state(x_quat):
#     # convert state x with attitude represented by quaternion
#     # to x_euler with attitude represented by Euler angles
#     return x_euler

# def quaternion_state(x_euler):
#     # convert state x_euler with attitude represented by Euler angles
#     # to x_quat with attitude represented by quaternions
#     return x_quat

# def f_euler(mav, x_euler, input):
#     # return 12x1 dynamics (as if state were Euler state)
#     return f_euler_

# def df_dx(mav, x_euler, input):
#     # take partial of f_euler with respect to x_euler
#     return A

# def df_du(mav, x_euler, delta):
#     # take partial of f_euler with respect to delta
#     return B

def dT_dVa(Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.1
    T = calc_Tp(Va,delta_t)
    T_eps = calc_Tp(Va + eps,delta_t)
    return (T_eps - T) / eps

def dT_ddelta_t(Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.01
    T = calc_Tp(Va,delta_t)
    T_eps = calc_Tp(Va,delta_t+eps)
    return (T_eps - T) / eps

def calc_Tp(Va,delta_t):
    Vin = MAV.V_max*delta_t
    a = MAV.rho*(MAV.D_prop**5)*MAV.C_Q0/(2*np.pi)**2
    b = MAV.rho*(MAV.D_prop**4)*MAV.C_Q1*Va/(2*np.pi) + MAV.KQ**2/MAV.R_motor
    c = MAV.rho*(MAV.D_prop**3)*MAV.C_Q2*Va**2 - MAV.KQ*Vin/MAV.R_motor + MAV.KQ*MAV.i0
    omega_p = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    T = (MAV.rho*(MAV.D_prop**4)*MAV.C_T0)*(omega_p**2) / (4*np.pi**2) + \
        (MAV.rho*(MAV.D_prop**3)*MAV.C_T1*Va*omega_p)/(2*np.pi) +\
        (MAV.rho*(MAV.D_prop**2)*MAV.C_T2*Va**2)
    return T
