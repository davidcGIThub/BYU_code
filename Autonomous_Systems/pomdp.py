#POMDP
import numpy as np
import matplotlib.pyplot as plt

#setup
p1 = np.linspace(0,1,100)
length = np.size(p1)
r = np.array([[-100 , 100], #r(x1,u1) r(x2,u1)
              [ 100 , -50], #r(x1,u2) r(x2,u2)
              [-1   , -1]]) #r(x1,u3) r(x2,u3)

p_x1n_u3 = np.array([0.2, 0.8]) #p(x1'|x1,u3) p(x1'|x2,u3)
p_x2n_u3 = np.array([0.8, 0.2]) #p(x2'|x1,u3) p(x2'|x2,u3)

p_z1_x = np.array([0.7, 0.3]) #p(z1|x1) p(z1|x2)
p_z2_x = np.array([0.3, 0.7]) #p(z2|x1) p(z2|x2)

#control
r_bu1 = p1*r[0,0] + (1-p1)*r[0,1] #p1*r(x1,u) + p2*r(x2,u)
r_bu2 = p1*r[1,0] + (1-p1)*r[1,1]
r_bu3 = p1*r[2,0] + (1-p1)*r[2,1]

V = np.maximum(r_bu1,r_bu2) #max r(b,u)
V = np.maximum(V,r_bu3)
u_map = np.zeros(length)
u_map[V == r_bu1] = 1
u_map[V == r_bu2] = 2
u_map[V == r_bu1] = 3

plt.plot(p1,r_bu1)
plt.plot(p1,r_bu2)
plt.plot(p1,r_bu3)
plt.plot(p1,V)
plt.show()

#sensing

p_z1 = p_z1_x[0]*p1 + p_z1_x[1]*(1-p1) #
p_z2 = p_z2_x[0]*p1 + p_z2_x[1]*(1-p1) #

p_x1_z1 = (p_z1_x[0] * p1)/p_z1
p_x1_z2 = (p_z2_x[0] * p1)/p_z2
#p_x2_z1 = (p_z1_x[1] * (1-p1))/p_z1
#p_x2_z2 = (p_z2_x[1] * (1-p1))/p_z2

r_bz1_u1 = p_x1_z1*r[0,0] + (1-p_x1_z1)*r[0,1] #r(b|z1,u)
r_bz1_u2 = p_x1_z1*r[1,0] + (1-p_x1_z1)*r[1,1]
r_bz1_u3 = p_x1_z1*r[2,0] + (1-p_x1_z1)*r[2,1]

V_z1 = np.maximum(r_bz1_u1,r_bz1_u2) #max r(bz1,u)
V_z1 = np.maximum(V_z1, r_bz1_u3)
u_map_z1 = np.zeros(length)
u_map_z1[V_z1 == r_bz1_u1] = 1
u_map_z1[V_z1 == r_bz1_u2] = 2
u_map_z1[V_z1 == r_bz1_u3] = 3

r_bz2_u1 = p_x1_z2*r[0,0] + (1-p_x1_z2)*r[0,1] #r(b|z2,u)
r_bz2_u2 = p_x1_z2*r[1,0] + (1-p_x1_z2)*r[1,1]
r_bz2_u3 = p_x1_z2*r[2,0] + (1-p_x1_z2)*r[2,1]

V_z2 = np.maximum(r_bz2_u1,r_bz2_u2) #max r(bz2,u)
V_z2 = np.maximum(V_z2, r_bz2_u3)
u_map_z2 = np.zeros(length)
u_map_z2[V_z2 == r_bz2_u1] = 1
u_map_z2[V_z2 == r_bz2_u2] = 2
u_map_z2[V_z2 == r_bz2



_u3] = 3

V_bar = V_z1*p_z1 + V_z2*p_z2


plt.plot(p1,V_bar)
plt.show()

plt.plot(p1,u_map_z1)
plt.plot(p1,u_map_z2)
plt.show()