#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:32:55 2021
EXTENDED KALMAN FILTER for STATE ESTIMATION
@author: tcallioglu
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s], tuple

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]


### INITIALIZES PARAMETERS
v_var = 0.01  # translation velocity variance  
om_var = 7  # rotational velocity variance 0.01
r_var = 0.01  # range measurements variance 0.1
b_var = 0.01  # bearing measurement variance 0.1

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance


# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x



### CORRECTION STEP for EKF
from numpy.linalg import inv
def measurement_update(lk, rk, bk, P_check, x_check):
    # x_check: [xk, yk, theta_k].T
    
    # 1. Compute measurement Jacobians Hk, Mk.
    x=x_check[0,0]
    y=x_check[0,1]
    theta=wraptopi(x_check[0,2])
    d=0
    
    star1=lk[0]-x-d*np.cos(theta)
    star2=lk[1]-y-d*np.sin(theta)
    den=np.sqrt(star1**2+star2**2)
    frac=star1**2+star2**2

    Hk=np.mat([[-star1/den,     -star2/den,     (star1*d*np.sin(theta)-star2*d*np.cos(theta))/den],
               [star2/frac,     -star1/frac,    -1-d*(np.sin(theta)*star2+np.cos(theta)*star1)/frac]])
    Mk=np.identity(2)

    # 2. Compute Kalman Gain
    Kk=P_check@Hk.T@inv(Hk@P_check@Hk.T+Mk@cov_y@Mk.T)

    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    yk=np.mat([den, wraptopi(np.arctan2(star2,star1)-theta)])
    ym=[rk,bk]
    x_check=x_check+(Kk@(ym-yk).T).T
    x_check[0,2]=wraptopi(x_check[0,2])
        
    # 4. Correct State covariance, P_check which means P_hat
    P_check=(np.identity(3)-Kk@Hk)@P_check

    return x_check, P_check


#### 5. Main Filter Loop #######################################################################
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)

    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    x_check = x_est[k-1,:]
    x_check=x_check.reshape((1,3))
    P_check=P_est[k-1,:];


    # 2. Motion model jacobian with respect to last state, Fk-1, Lk-1.
    theta=wraptopi(x_check[0,2])
    Fu=np.mat([[np.cos(theta), 0],
               [np.sin(theta), 0],
               [0,1]])
    
    inputC=np.mat([v[k-1],om[k-1]])     # om[k-1] = wk = angular, rotational velocity, not process noise
    x_check=x_check+delta_t*(Fu@inputC.T).T
    x_check[0,2]=wraptopi(x_check[0,2])

    Fkm=np.mat([[1,0, -delta_t*np.sin(theta)*v[k-1]],[0,1, delta_t*np.cos(theta)*v[k-1]], [0,0,1]])
    Lkm=delta_t*Fu

    # 3. Motion model jacobian with respect to noise. Because of wk=0 (process noise), there is nothing to do. 


    # 4. Propagate uncertainty
    P_check=Fkm@P_check@Fkm.T+Lkm@Q_km@Lkm.T
    
    
    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0,0]
    x_est[k, 1] = x_check[0,1]
    x_est[k, 2] = wraptopi(x_check[0,2])
    P_est[k, :, :] = P_check
    



e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()