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

def Euler2RotationMatrix(phi,theta,psi):
    R = np.array([[np.cos(theta)*np.cos(psi) ,  #rotation matrix body to vehicle frame, or rotate from vehicle to new location
                    np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi),
                    np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
                [np.cos(theta)*np.sin(psi),
                    np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi),
                    np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
                [-np.sin(theta),
                    np.sin(phi)*np.cos(theta),
                    np.cos(phi)*np.cos(theta)]])
    return R

def Quaternion2RotationMatrix(quat):
    if len(quat) != 4:
        print("Error: Quaternion size not valid")
        return False
    quaternion = quat / np.linalg.norm(quat)
    r = quaternion[0]
    i = quaternion[1]
    j = quaternion[2]
    k = quaternion[3]
    R = np.array([[1-2*(j*j + k*k) , 2*(i*j-k*r) , 2*(i*k + j*r)],
                    [2*(i*j + k*r) , 1 - 2*(i*i + k*k) , 2*(j*k - i*r)],
                    [2*(i*k - j*r) , 2*(j*k + i*r) , 1-2*(i*i + j*j)]])
    return R