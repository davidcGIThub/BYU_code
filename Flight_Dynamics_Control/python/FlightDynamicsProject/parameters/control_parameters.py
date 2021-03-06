import sys
sys.path.append('..')
import numpy as np
import project.model_coef as TF
import parameters.aerosonde_parameters as MAV

Va_trim = TF.Va_trim

delta_a_trim = TF.u_trim[1,0]
delta_r_trim = TF.u_trim[2,0]
beta_trim = TF.beta_trim
u_trim = TF.x_trim[3,0]
v_trim = TF.x_trim[4,0]
w_trim = TF.x_trim[5,0]
p_trim = TF.x_trim[9,0]
q_trim = TF.x_trim[10,0]
r_trim = TF.x_trim[11,0]

#lateral state space model coeficients
Yv = (MAV.rho*MAV.S_wing*MAV.b*v_trim/(4*MAV.mass*Va_trim)) * (MAV.C_Y_p*p_trim + MAV.C_Y_r*r_trim) \
    + (MAV.rho*MAV.S_wing*v_trim/MAV.mass) * (MAV.C_Y_0 + MAV.C_Y_beta*beta_trim + \
        MAV.C_Y_delta_a*delta_a_trim + MAV.C_Y_delta_r*delta_r_trim) + \
            (MAV.rho*MAV.S_wing*MAV.C_Y_beta/(2*MAV.mass))*np.sqrt(u_trim**2 + w_trim**2)
Yr = -u_trim + MAV.rho*Va_trim*MAV.S_wing*MAV.b *MAV.C_Y_r / (4*MAV.mass)
Ydelr = (MAV.rho*Va_trim**2)*MAV.S_wing*MAV.C_Y_delta_r / (2*MAV.mass)
Nv = MAV.rho*MAV.S_wing*(MAV.b**2)*v_trim/(4*Va_trim) * (MAV.C_r_p*p_trim + MAV.C_r_r*r_trim) \
    + MAV.rho*MAV.S_wing*MAV.b*v_trim * \
        (MAV.C_r_0 + MAV.C_r_beta*beta_trim + MAV.C_r_delta_a*delta_a_trim + MAV.C_r_delta_r*delta_r_trim) \
            + (MAV.rho*MAV.S_wing*MAV.b*MAV.C_r_beta/2)*np.sqrt(u_trim**2 + w_trim**2)
Nr = -MAV.gamma1*q_trim + MAV.rho*Va_trim*MAV.S_wing*(MAV.b**2)*MAV.C_r_r/4
Ndelr = MAV.rho*(Va_trim**2)*MAV.S_wing*MAV.b*MAV.C_r_delta_r/2


#----------roll loop-------------
# get transfer function data for delta_a to phi
delta_a_max = 0.78 #45 deg
e_roll_max = 1.57 #3.14
wn_roll = np.sqrt( np.abs(TF.a_phi2) * delta_a_max/e_roll_max) #natural frequency
zeta_roll = .707 # damping ratio
roll_kp = (wn_roll**2)/TF.a_phi2 #proportional gain # 5.625
roll_kd = (2.0*zeta_roll*wn_roll - TF.a_phi1) / TF.a_phi2 #derivative gain #0.2417

#----------course loop-------------
roll_max = 0.52 #30 deg
Vg = Va_trim #assuming zero airspeed, and average speed = Va_trim
W_course = 8
zeta_course = .707
wn_course = wn_roll/W_course
course_kp = 2*zeta_course*wn_course*Vg/MAV.gravity #0.7935
course_ki = (wn_course**2)*Vg/MAV.gravity #0.123

#----------yaw damper-------------
wn_yaw = np.sqrt(Yv*Nr-Yr*Nv)
yaw_damper_tau_r = 10/wn_yaw #0.5
yaw_damper_kp = -(Nr*Ndelr + Ydelr*Nv)/(Ndelr**2) + np.sqrt( ((Nr*Ndelr + Ydelr*Nv)/Ndelr**2)**2  \
    - ( (Yv**2 + Nr**2 + 2*Yr*Nv)/(Ndelr**2) ) ) #0.3

#----------pitch loop-------------
delta_e_max = 0.78
e_pitch_max = 0.08
wn_pitch = np.sqrt(TF.a_theta2 + np.abs(TF.a_theta3)*delta_e_max/e_pitch_max)
zeta_pitch = 0.707
pitch_kp = ((wn_pitch**2) - TF.a_theta2)/TF.a_theta3 #50.84
pitch_kd = (2*zeta_pitch*wn_pitch-TF.a_theta1)/TF.a_theta3 #-0.828116
K_theta_DC = pitch_kp*TF.a_theta3/(wn_pitch**2) # 0.9483 

#----------altitude loop-------------
pitch_max = 0.52 #30 deg
W_altitude = 20
zeta_altitude = 0.707
wn_altitude = wn_pitch/W_altitude
altitude_kp = 2*zeta_altitude*wn_altitude / (K_theta_DC*Va_trim) #0.3247
altitude_ki = (wn_altitude**2) / (K_theta_DC*Va_trim) #0.51034
altitude_zone = [30,400] # moving saturation limit around current altitude

#---------airspeed hold using throttle---------------
throttle_max = 1.0
wn_airspeed_throttle = 5*wn_altitude
zeta_airspeed_throttle = 0.707
airspeed_throttle_kp = (2*zeta_airspeed_throttle*wn_airspeed_throttle - TF.a_V1) / TF.a_V2 #0.02004
airspeed_throttle_ki = (wn_airspeed_throttle**2)/TF.a_V2 #5.0
