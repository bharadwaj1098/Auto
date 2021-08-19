import numpy as np
import pandas as pd
from scipy.integrate import odeint
import random
from random import randint
from pdb import set_trace  
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
        self.sensor_spring = dic['sensor_spring'] #K_T
        self.sw_mass = dic['sw_mass']
        self.state = dic['state']
        self.Theta_h = Theta_h
        self.Theta_a = Theta_a
        self.G = dic['G'] # r_s/r_m
        self.T_v = dic['T_v']
        self.J_sm = dic['J_sm'] #[j_s, j_m]
        self.args_dict_h = dic['args_dict_h']
        self.args_dict_a = dic['args_dict_a']
        self.Theta = self.Theta_dict() 
        self.diff_theta_h = self.Diff_theta(self.Theta_h, self.Theta['H_time'], self.args_dict_h['amplitude'], self.args_dict_h['omega'] )
        self.diff_theta_a = self.Diff_theta(self.Theta_a, self.Theta['A_time'], self.args_dict_a['amplitude'], self.args_dict_a['omega'] )
        self.flag = 1 if len(self.Theta['H_time']) == len(self.Theta['A_time']) else 0
        self._vector = odeint(self.callback_func, self.state, self.Theta['H_time'], mxstep = 50000000) if self.flag==1 else [0,0,0,0]
        self.acc_sw, self.acc_s, self.error = self.auto_acc() 

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
        
        if self.dic['neg'] == True:
            Theta['A'] = []
            for i in range(len(self.mds_a.force_list)):
                Theta['A'].append( -1 * self.mds_a.force_list[i])  
        else:
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

            if self.dic['non_linear'] != 'yes': 
                delta_k_h = self.dic['delta_K_H'] * (self.Theta['H'][T] - self._vector[:,0][T] )
                delta_k_a = self.G * self.dic['delta_K_A'] * (self.Theta['A'][T] - (self.G * self._vector[:,2][T] ) )

            elif self.dic['non_linear'] == 'yes':
                 delta_k_h = ((self.Theta['H_spring'][T] ** 3) - self.Theta['H_spring'][T] ) * (self.Theta['H'][T] - self._vector[:,0][T] ) 
                 delta_k_a = self.G * ((self.Theta['A_spring'][T] ** 3) - self.Theta['A_spring'][T] ) \
                      * (self.Theta['A'][T] - (self.G * self._vector[:,2][T] ) )

            delta_b_h = self.dic['delta_B_H'] * (self.diff_theta_h[T] - self._vector[:,1][T] )
            delta_b_a = self.G * self.dic['delta_B_A'] * (self.diff_theta_a[T] - (self.G * self._vector[:,3][T] ) )

            a = (delta_b_h + delta_k_h) * (1/j1)
            b = (delta_b_a + delta_k_a) * deno  
            error.append( [a, b] )
        return acc_h, acc_a, np.asarray(error) 

    def dataframe(self):
        D = {
            'time_step_H' : self.Theta['H_time'],
            'time_step_A' : self.Theta['A_time'],
            'theta_H' : self.Theta['H'],
            'theta_A' : self.Theta['A'],
            'H_spring' : self.Theta['H_spring'],
            'A_spring' : self.Theta['A_spring'],
            'H_damp' : self.Theta['H_Damp'],
            'A_damp' : self.Theta['A_Damp'],
            'H_mass' : self.Theta['H_mass'],
            'A_mass' : self.Theta['A_mass'],
            'disp_sw': self._vector[:,0],
            'disp_s': self._vector[:, 2],
            'vel_sw': self._vector[:, 1],
            'vel_s': self._vector[:, 3],
            'acc_sw': self.acc_sw,
            'acc_s': self.acc_s, 
            'error_1': self.error[:, 0], 
            'error_2': self.error[:, 1] 
        }

        df = pd.DataFrame(D)
        return df

#this class is make the theta_H == theta_A and both t_h, t_a are according to vahid's paper

class haptic_actual:
    def __init__(self, dic):
        self.dic = dic
        self.time_array = np.arange(0, dic['time'] + dic['increment'], dic['increment'])
        self.args_dict_a = dic['args_dict_a']
        self.args_dict_h = dic['args_dict_h']
        self.theta_h, self.diff_theta_h = self.theta_h_a()
        self.theta_a, self.diff_theta_a = self.theta_h, self.diff_theta_h
        self.Auto_theta = self.Auto()
        self.data_list = self.Human()

    def theta_h_a(self):
        '''
        T1 = 10, T2 = 25, T3 = 30
        W = 0.3 
        '''
        t = Symbol('t')
        t_h = [] 
        for i in self.time_array:
            if i <= 10:
                t_h.append(0) 
            elif i <= 35 and i > 10:
                f = (0.3/2) * (cos((np.pi/32)*(t-35)) + 1) 
                f = lambdify(t, f)
                t_h.append( f(i) )
            elif i <= 65 and i > 35:
                t_h.append(0.3)
            elif i<=90 and i>65:
                f = (0.3/2) * (cos((np.pi/32)*(t-65)) + 1) 
                f = lambdify(t, f)
                t_h.append( f(i) ) 
            elif i > 90:
                t_h.append(0)

        diff_t = list(np.gradient(t_h, self.dic['increment'] ))
        return t_h, diff_t

    def Auto(self):
        Theta = {}
        self.mds_a = mds(time = self.dic['time'], increment = self.dic['increment'], state=[0,0], 
                    no_mass = self.args_dict_a['no_mass'], range_mass=self.args_dict_a['range_mass'], 
                    no_spring= self.args_dict_a['no_spring'], range_spring = self.args_dict_a['range_spring'], 
                    no_damp = self.args_dict_a['no_damp'], range_damp=self.args_dict_a['range_damp'], uniform=1)

        Theta['A_mass'] = self.mds_a.mass
        Theta['A_spring'] = self.mds_a.spring 
        Theta['A_Damp'] = self.mds_a.damp
        Theta['A_time']  = self.time_array
        Theta['A_Theta'] = self.theta_a
        Theta['A_Theta_diff'] = self.diff_theta_a 

        return Theta 
    
    def Human(self):
        data_list = []

        for i in range(self.dic['number_of_runs']):
            
            Theta = {}

            mds_h = mds(time = self.dic['time'], increment = self.dic['increment'], state=[0,0], 
                    no_mass = self.args_dict_h['no_mass'], range_mass=self.args_dict_h['range_mass'], 
                    no_spring= self.args_dict_h['no_spring'], range_spring = self.args_dict_h['range_spring'], 
                    no_damp = self.args_dict_h['no_damp'], range_damp=self.args_dict_h['range_damp'], uniform=1)

            Theta['H_mass'] = mds_h.mass
            Theta['H_spring'] = mds_h.spring 
            Theta['H_Damp'] = mds_h.damp
            Theta['H_time']  = self.time_array
            Theta['H_Theta'] = self.theta_h
            Theta['H_Theta_diff'] = self.diff_theta_h 

            def callback_func(state, t): 
                T_h = int(t * mds_h.product) 
                dx1dt = state[1] 
                b_h = self.theta_h[T_h] * \
                    (self.diff_theta_h[T_h] - \
                        state[1]) 
                k_h = Theta['H_spring'][T_h] * (Theta['H_Theta'][T_h] - state[0]) 
                K_t = self.dic['sensor_spring'] * (state[0] - state[2]) 
                j1 = self.dic['sw_mass'] + Theta['H_mass'][T_h] 
                dx2dt = (b_h + k_h - K_t) * (1/j1)
                dx3dt = state[3] 
                b_a = self.dic['G'] * self.Auto_theta['A_Damp'][T_h] * (self.diff_theta_a[T_h] - (self.dic['G'] * state[3]))
                k_a = self.dic['G'] * self.Auto_theta['A_spring'][T_h] * (self.Auto_theta['A_Theta'][T_h] - (self.dic['G'] * state[2])) 
                dx4dt = (b_a + k_a + K_t + self.dic['T_v']) * ( 1 / ((self.dic['G'] **2) * self.dic['J_sm'][1] + self.dic['J_sm'][0]) ) 
                dxdt = [dx1dt, dx2dt, dx3dt, dx4dt] 
                return dxdt 

            Human_vector = odeint(callback_func, \
                self.dic['state'], \
                    self.time_array, \
                        mxstep = 50000000)

            def auto_acc():
                acc_h , acc_a, error = [], [], []
                for i in Theta['H_time']:
                    T = int(i * mds_h.product)
                    b_h = Theta['H_Damp'][T] * (Theta['H_Theta'][T] - Human_vector[:,1][T] )
                    k_h = Theta['H_spring'][T] * (Theta['H_Theta'][T] - Human_vector[:,0][T] ) 
                    K_t = self.dic['sensor_spring'] * (Human_vector[:,0][T] - Human_vector[:,2][T] ) 
                    j1 = self.dic['sw_mass'] + Theta['H_mass'][T]  
                    acc_h.append( (b_h + k_h - K_t) * (1/j1) )

                    b_a = self.dic['G'] * self.Auto_theta['A_Damp'][T] * (self.Auto_theta['A_Theta_diff'][T] - (self.dic['G'] * Human_vector[:,3][T] ) )
                    k_a = self.dic['G'] * self.Auto_theta['A_spring'][T] * (self.Auto_theta['A_Theta'][T] - (self.dic['G'] * Human_vector[:,2][T] ) ) 
                    deno = ( 1 / ( (self.dic['G'] **2) * self.dic['J_sm'][1] + self.dic['J_sm'][0] ) ) 
                    num = (b_a + k_a + K_t + self.dic['T_v']) 
                    acc_a.append(num * deno) 

                    delta_b_h = self.dic['delta_B_H'] * (self.diff_theta_h[T] - Human_vector[:,1][T] )
                    delta_k_h = self.dic['delta_K_H'] * (Theta['H_Theta'][T] - Human_vector[:,0][T] ) 
                    a = (delta_b_h + delta_k_h) * (1/j1)

                    delta_b_a = self.dic['G'] * self.dic['delta_B_A'] * (self.Auto_theta['A_Theta_diff'][T] - (self.dic['G'] * Human_vector[:,3][T] ) ) 
                    delta_k_a = self.dic['G'] * self.dic['delta_K_A'] * (self.Auto_theta['A_Theta'][T] - (self.dic['G'] * Human_vector[:,2][T] ) )
                    b = (delta_b_a + delta_k_a) * deno  
                    error.append( [a, b] )

                return acc_h, acc_a, np.asarray(error)
            
            Acc_sw, Acc_s, error = auto_acc()

            D = {
                'sim_number' : i+1,
                'time_step_H' : self.time_array,
                'time_step_A' : self.time_array,
                'theta_H' : Theta['H_Theta'],
                'theta_A' : self.Auto_theta['A_Theta'],
                'H_spring' : Theta['H_spring'],
                'A_spring' : self.Auto_theta['A_spring'],
                'H_damp' : Theta['H_Damp'],
                'A_damp' : self.Auto_theta['A_Damp'],
                'H_mass' : Theta['H_mass'],
                'A_mass' : self.Auto_theta['A_mass'],
                'disp_sw': Human_vector[:,0],
                'disp_s': Human_vector[:, 2],
                'vel_sw': Human_vector[:, 1],
                'vel_s': Human_vector[:, 3],
                'acc_sw': Acc_sw,
                'acc_s': Acc_s, 
                'error_1': error[:, 0], 
                'error_2': error[:, 1]
            }

            data_list.append( pd.DataFrame(D) ) 
        
        return data_list