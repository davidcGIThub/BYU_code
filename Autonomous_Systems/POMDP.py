#POMDP
import numpy as np
import matplotlib.pyplot as plt

#setup
p1 = np.linspace(0,1,100)
length = np.size(p1)
r = np.array([[-100 , 100], #r(x1,u1) r(x2,u1)
              [ 100 , -50], #r(x1,u2) r(x2,u2)
              [-1   , -1]]) #r(x1,u3) r(x2,u3)
cost_u3 = 1.0
p_x1n_u3 = np.array([0.2, 0.8]) #p(x1'|x1,u3) p(x1'|x2,u3)

p_z1_x = np.array([0.7, 0.3]) #p(z1|x1) p(z1|x2)
p_z2_x = np.array([0.3, 0.7]) #p(z2|x1) p(z2|x2)
horizon = 20
image_num = 1

#control calculations
r_bu1 = p1*r[0,0] + (1-p1)*r[0,1] #p1*r(x1,u) + p2*r(x2,u)
r_bu2 = p1*r[1,0] + (1-p1)*r[1,1]
r_bu3 = p1*r[2,0] + (1-p1)*r[2,1]
image_num = image_num-1
#Initial Conditions
V_bar = np.maximum(r_bu1,r_bu2) #max r(b,u)
V_bar = np.maximum(V_bar,r_bu3)
u_map = np.zeros(length)
u_map[V_bar == r_bu1] = 1
u_map[V_bar == r_bu2] = 2
u_map[V_bar == r_bu1] = 3

r_b = np.array([r_bu1 , r_bu2 , r_bu3])
for i in range(0,horizon):
    #sense
    p1n = p_x1n_u3[0]*p1 + p_x1n_u3[1] * (1-p1) # this is techically a prediction step

    p_z1 = p_z1_x[0]*p1n + p_z1_x[1]*(1-p1n) # p(z1)
    p_x1_z1 = (p_z1_x[0] * p1n)/p_z1 #p(x1|z1) or p1n'z1

    p_z2 = p_z2_x[0]*p1n + p_z2_x[1]*(1-p1n) # p(z2)
    p_x1_z2 = (p_z2_x[0] * p1n)/p_z2 #p(x1|z2) or p1n'z2

    Val = (p_x1_z1*r[0,0] + (1-p_x1_z1)*r[0,1])*p_z1 + (p_x1_z2*r[0,0] + (1-p_x1_z2)*r[0,1])*p_z2
    u_map = np.zeros(length)
    r_all = np.array([[]])
    rb_all = np.array([[]])
    num = np.size(r,0)
    for j in range(0,num):
        for k in range(0,num):
            rb1_temp = p_x1_z1*r[j,0] + (1-p_x1_z1)*r[j,1] #r(b|z1,u)
            rb2_temp = p_x1_z2*r[k,0] + (1-p_x1_z2)*r[k,1] #r(b|z2,u)
            rb_temp = rb1_temp * p_z1 + rb2_temp * p_z2
            Val = np.maximum(Val, rb_temp)
            #prediction steps
            u_map[Val == rb_temp] = num*j+k
            r2 = rb_temp[0] - cost_u3
            r1 = rb_temp[length-1] - cost_u3
            if j==k==0:
                r_all = np.array([[r1,r2]])
                rb_all = np.array([rb_temp])
            else:
                r_all = np.concatenate((r_all,np.array([[r1,r2]])))
                rb_all = np.concatenate((rb_all,[rb_temp]))

    if i == 0 == image_num:  # first time step
        for q in range(0,np.size(rb_all,0)):
            plt.plot(p1,rb_all[q],'--')
        plt.plot(p1,Val,'k')
        plt.xlim(0,1)
        plt.ylim(-100,100)
        plt.xlabel("p1 (belief(x1))")
        plt.ylabel("V (value function)")
        plt.title('Horizon Step = ' + str(image_num+1))
        plt.show()

    else:
        ind_max = np.unique(u_map).astype(int)
        r = np.array([r[0] , r[1]])
        r = np.concatenate((r, r_all[ind_max]))

        cost_func = cost_u3 - p1*cost_u3
        V_bar = Val - cost_func
        V_bar = np.maximum(V_bar, r_bu1)
        V_bar = np.maximum(V_bar, r_bu2)
        u_map[V_bar == r_bu1] = 1
        u_map[V_bar == r_bu2] = 2
        if i == image_num:
            for q in range(0,np.size(rb_all,0)):
                plt.plot(p1,rb_all[q],'--')
            plt.plot(p1,V_bar,'k')
            plt.xlim(0,1)
            plt.ylim(-100,100)
            plt.xlabel("p1 (belief(x1))")
            plt.ylabel("V (value function)")
            plt.title('Horizon Step = ' + str(image_num+1))
            plt.show()
'''
    p_z1 = p_z1_x[0]*p1n + p_z1_x[1]*(1-p1n) # p(z1)
    p_x1_z1 = (p_z1_x[0] * p1n)/p_z1 #p(x1|z1) or p1n'z1
    V_z1 = (p_x1_z1*r[0,0] + (1-p_x1_z1)*r[0,1]) * 1.0
    u_map_z1 = np.zeros(length)
    for j in range(0,np.size(r_b,0)):
        rb_temp = p_x1_z1*r[j,0] + (1-p_x1_z1)*r[j,1] #r(b|z1,u)
        V_z1 = np.maximum(rb_temp,V_z1) #max r(bz1,u)
        u_map_z1[V_z1 == rb_temp] = j+1

    p_z2 = p_z2_x[0]*p1n + p_z2_x[1]*(1-p1n) # p(z2)
    p_x1_z2 = (p_z2_x[0] * p1n)/p_z2 #p(x1|z2) or p1n'z2
    V_z2 = p_x1_z2*r[0,0] + (1-p_x1_z2)*r[0,1]
    u_map_z2 = np.zeros(length)
    for j in range(0,np.size(r_b,0)):
        rb_temp = p_x1_z2*r[j,0] + (1-p_x1_z2)*r[j,1] #r(b|z2,u)
        V_z2 = np.maximum(V_z2,rb_temp) #max r(bz2,u)
        u_map_z2[V_z2 == rb_temp] = j+1
'''