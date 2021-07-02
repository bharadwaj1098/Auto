import numpy as np
from scipy.integrate import odeint
import random
from random import randint 
from sympy import *
from spring_damp import mass_damp_spring as mds
import matplotlib.pyplot as plt
random.seed(49)

class haptic_class:
    '''
    Input state_vector:-
    state[0] = Theta_sw disp
    state[1] = Theta_sw angular vel
    state[2] = theta_s disp
    state[3] = theta_s angular vel

    G = gear ratio of steering_wheel, 1/5 or 1.5
    T_v = 0 
    sensor_spring = spring of sensor (K_t)
    J_sm[0] = J_s 
    J_sm[1] = J_m  
    '''
    theta_state= [0]
    def __init__(self, dic, Theta_h, Theta_a):
        self.dic = dic
        self.time = dic['time']
        self.sensor_spring = dic['sensor_spring']
        self.sw_mass = dic['sw_mass']
        self.state = dic['state']
        self.Theta_h = Theta_h
        self.Theta_a = Theta_a
        self.G = dic['G'] 
        self.T_v = dic['T_v']
        self.J_sm = dic['J_sm'] 
        self.args_dict_h = dic['args_dict_h']
        self.args_dict_a = dic['args_dict_a']
        self.Theta = self.Theta_dict() 
        self.diff_theta_h = self.Diff_theta(self.Theta_h, self.Theta['H_time'], self.args_dict_h['amplitude'], self.args_dict_h['omega'] )
        self.diff_theta_a = self.Diff_theta(self.Theta_a, self.Theta['A_time'], self.args_dict_a['amplitude'], self.args_dict_a['omega'] )
        self.flag = 1 if len(self.Theta['H_time']) == len(self.Theta['A_time']) else 0
        self._vector = odeint(self.callback_func, self.state, self.Theta['H_time'], mxstep = 50000000) if self.flag==1 else [0,0,0,0]
        self.acc_h, self.acc_a, self.error = self.auto_acc() 

    def Theta_dict(self):
        Theta = {}
        self.mds_h = mds(time = self.time, increment = 0.01, state=self.state, #state_actually doesn't matter for sprring_damp_class
                    force = self.Theta_h, amplitude= self.args_dict_h['amplitude'], omega=self.args_dict_h['omega'], 
                    no_mass = self.args_dict_h['no_mass'], range_mass=self.args_dict_h['range_mass'], 
                    no_spring= self.args_dict_h['no_spring'], range_spring = self.args_dict_h['range_spring'], 
                    no_damp = self.args_dict_h['no_damp'], range_damp=self.args_dict_h['range_damp'], uniform=1) 
        
        self.mds_a = mds(time = self.time, increment = 0.01, state=self.state,
                    force = self.Theta_a, amplitude= self.args_dict_a['amplitude'], omega=self.args_dict_a['omega'], 
                    no_mass = self.args_dict_a['no_mass'], range_mass=self.args_dict_a['range_mass'], 
                    no_spring= self.args_dict_a['no_spring'], range_spring = self.args_dict_a['range_spring'], 
                    no_damp = self.args_dict_a['no_damp'], range_damp=self.args_dict_a['range_damp'], uniform=1)

        Theta['H'] = self.mds_h.force_list #Theta_H
        Theta['H_mass'] = self.mds_h.mass 
        Theta['H_spring'] = self.mds_h.spring 
        Theta['H_Damp'] = self.mds_h.damp 
        Theta['H_time']  = self.mds_h.time_array

        Theta['A'] = self.mds_a.force_list 
        Theta['A_mass'] = self.mds_a.mass 
        Theta['A_spring'] = self.mds_a.spring 
        Theta['A_Damp'] = self.mds_a.damp
        Theta['A_time']  = self.mds_a.time_array

        return Theta

    def Diff_theta(self, Theta, time_array, a, w):
        diff_theta = []
        t = Symbol('t')
        if a is not None and w is not None:
            y = a * Theta(w * t)
        else :
            y = Theta(t)
        dt = diff(y, t)
        ddt = lambdify(t, dt)
        for i in time_array:
            diff_theta.append(ddt(i)) 
        return diff_theta 

    def callback_func(self, state, t): 
        T_h = int(t * self.mds_h.product) 
        dx1dt = state[1] 
        b_h = self.Theta['H_Damp'][T_h] * (self.diff_theta_h[T_h] - state[1]) 
        k_h = self.Theta['H_spring'][T_h] * (self.Theta['H'][T_h] - state[0]) 
        K_t = self.sensor_spring * (state[0] - state[2]) 
        j1 = self.sw_mass + self.Theta['H_mass'][T_h] 
        dx2dt = (b_h + k_h - K_t) * (1/j1)
        dx3dt = state[3] 
        b_a = self.G * self.Theta['A_Damp'][T_h] * (self.diff_theta_a[T_h] - (self.G * state[3]))
        k_a = self.G * self.Theta['A_spring'][T_h] * (self.Theta['A'][T_h] - (self.G * state[2])) 
        dx4dt = (b_a + k_a + K_t + self.T_v) * ( 1 / ((self.G **2) * self.J_sm[1] + self.J_sm[0]) ) 
        dxdt = [dx1dt, dx2dt, dx3dt, dx4dt] 
        return dxdt 

    def auto_acc(self):
        acc_h , acc_a, error = [], [], []
        for i in self.Theta['A_time']:
            T = int(i * self.mds_a.product)
            b_h = self.Theta['H_Damp'][T] * (self.diff_theta_h[T] - self._vector[:,1][T] )
            k_h = self.Theta['H_spring'][T] * (self.Theta['H'][T] - self._vector[:,0][T] ) 
            K_t = self.sensor_spring * (self._vector[:,0][T] - self._vector[:,2][T] ) 
            j1 = self.sw_mass + self.Theta['H_mass'][T]  
            acc_h.append( (b_h + k_h - K_t) * (1/j1) )

            b_a = self.G * self.Theta['A_Damp'][T] * (self.diff_theta_a[T] - (self.G * self._vector[:,3][T] ) )
            k_a = self.G * self.Theta['A_spring'][T] * (self.Theta['A'][T] - (self.G * self._vector[:,2][T] ) ) 
            deno = ( 1 / ( (self.G **2) * self.J_sm[1] + self.J_sm[0] ) ) 
            num = (b_a + k_a + K_t + self.T_v) 
            acc_a.append(num * deno) 

            delta_b_h = self.dic['delta_B_H'] * (self.diff_theta_h[T] - self._vector[:,1][T] )
            delta_k_h = self.dic['delta_K_H'] * (self.Theta['H'][T] - self._vector[:,0][T] ) 
            a = (delta_b_h + delta_k_h) * (1/j1)

            delta_b_a = self.G * self.dic['delta_B_A'] * (self.diff_theta_a[T] - (self.G * self._vector[:,3][T] ) ) 
            delta_k_a = self.G * self.dic['delta_K_A'] * (self.Theta['A'][T] - (self.G * self._vector[:,2][T] ) )
            b = (delta_b_a + delta_k_a) * deno  
            error.append( [a, b] )

        return acc_h, acc_a, np.asarray(error)
