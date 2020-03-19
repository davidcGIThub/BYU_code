"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
import copy as cp
import parameters.simulation_parameters as SIM
from chap4.mav_dynamics import mav_dynamics


def compute_trim(mav, delta, Va, gamma, R = 10000.0):
    # define initial state and input
    state0 = cp.deepcopy(mav._state) #[[pn],[pe],[pd],[u],[v],[w],[e0],[e1],[e2],[e3],[p],[q],[r]]
    delta0 = cp.deepcopy(delta) # [delta_e, delta_a, delta_r, delta_t]
    x0 = np.concatenate((state0, delta0), axis=0)
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2, #(u^2 + v^2 + w^2 = Va^2) magnitude of velocity vector is Va
                                x[4],  # (v=0), force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # (e0^2 + e1^2 + e2^2 + e3^3 = 1) force quaternion to be unit length
                                x[7],  # (e1=0)  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # (e3=0)
                                x[10],  # (p=0) - angular rates should all be zero
                                x[11],  # (q=0)
                                x[12],  # (r=0)
                                ]),
             #Jacobian of the above function with respect to each variable x
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma),
                   constraints=cons, options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = np.array([res.x[13:17]]).T
    print('trim_state=', trim_state.T)
    print('trim_input=', trim_input.T)
    return trim_state, trim_input

# objective function to be minimized
def trim_objective_fun(x, mav, Va, gamma, R = 100000):
    x_dot_star = np.array([[Va*np.sin(gamma)],      #h_dot*
                        [0],                    #u_dot*
                        [0],                    #v_dot*
                        [0],                    #w_dot*
                        [0],                    #phi_dot*
                        [0],                    #theta_dot*
                        [(Va/R)*np.cos(gamma)], #psi_dot*
                        [0],                    #p_dot*
                        [0],                    #q_dot*
                        [0]])                   #r_dot*

    mav_temp = mav_dynamics(SIM.ts_simulation)
    state = np.array([x[0:13]]).T
    delta = np.array([x[13:17]]).T
    mav_temp._state = state
    forces_moments = mav_temp._forces_moments(delta)
    x_dot = mav_temp._derivatives(state,forces_moments)
    J = np.linalg.norm(x_dot_star-x_dot[3:13])**2
    return J

