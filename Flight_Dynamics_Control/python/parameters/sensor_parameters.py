import sys
sys.path.append('..')
import numpy as np

# -------- Accelerometer --------
accel_sigma = 0.0025*9.8  # standard deviation of accelerometers in m/s^2

# -------- Rate Gyro --------
gyro_x_bias = 0  # np.radians(5*np.random.uniform(-1, 1))  # bias on x_gyro
gyro_y_bias = 0  # np.radians(5*np.random.uniform(-1, 1))  # bias on y_gyro
gyro_z_bias = 0  # np.radians(5*np.random.uniform(-1, 1))  # bias on z_gyro
gyro_sigma = np.radians(0.13)  # standard deviation of gyros in rad/sec

# -------- Pressure Sensor(Altitude) -------- absolute pressure
static_pres_sigma = 0.01*1000  # standard deviation of static pressure sensors in Pascals
static_pres_beta = 0 #0.01*100 * np.random.uniform(-1,1)

# -------- Pressure Sensor (Airspeed) -------- differential pressure
diff_pres_sigma = 0.002*1000  # standard deviation of diff pressure sensor in Pascals
diff_pres_beta = 0 #0.001*100 * np.random.uniform(-1,1)

# -------- Magnetometer --------
mag_beta = np.radians(1.0)
mag_sigma = np.radians(0.03)
mag_incl = np.radians(-66) #magnetic inclination
mag_decl = np.radians(12.5)

# -------- GPS --------
ts_gps = 1.0 #sample time
k_gps = 1. / 1100.  # 1 / s #time constant of the process
gps_n_sigma = 0.21
gps_e_sigma = 0.21
gps_h_sigma = 0.40
#assuming the uncertainty in the north and east directions same
gps_Vg_sigma = 0.05
#gps_course_sigma = gps_Vg_sigma / Vg