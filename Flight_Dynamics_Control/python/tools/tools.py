#tools
import numpy as np

def Quaternion2Euler(quatArray):
    e0 = quatArray[0]
    e1 = quatArray[1]
    e2 = quatArray[2]
    e3 = quatArray[3]

    roll = np.arctan2(2*(e0*e1+e2*e3), (e0**2+e3**2-e1**2-e2**2))
    pitch = np.arcsin(2*(e0*e2-e1*e3))
    yaw = np.arctan2(2*(e0*e3+e1*e2),(e0**2+e1**2-e2**2-e3**2))

    return roll,pitch,yaw

def Euler2Quaternion(roll,pitch,yaw):

    e0 = np.cos(yaw/2)*np.cos(pitch/2)*np.cos(roll/2) + np.sin(yaw/2)*np.sin(pitch/2)*np.sin(roll/2)
    e1 = np.cos(yaw/2)*np.cos(pitch/2)*np.sin(roll/2) - np.sin(yaw/2)*np.sin(pitch/2)*np.cos(roll/2)
    e2 = np.cos(yaw/2)*np.sin(pitch/2)*np.cos(roll/2) + np.sin(yaw/2)*np.cos(pitch/2)*np.sin(roll/2)
    e3 = np.sin(yaw/2)*np.cos(pitch/2)*np.cos(roll/2) - np.cos(yaw/2)*np.sin(pitch/2)*np.sin(roll/2)

    return np.array([e0,e1,e2,e3])