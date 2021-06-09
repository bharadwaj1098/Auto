import numpy as np
from scipy.integrate import odeint
import pandas as pd
import random
from random import randint 
from sympy import *
import matplotlib.pyplot as plt
random.seed(49)

class force_time:
    def __init__(self, time, increment,state,force= None, amplitude=None, omega=None):
        self.state=state
        self.increment = increment
        self.product = 1 / (self.increment)
        self.time_array = np.arange(0, time + self.increment, self.increment)
        self.force = force
        self.amplitude = amplitude 
        self.omega = omega
        self.force_list = self._force_list()

    def _force_list(self):
        _list = []
        for i in self.time_array:
            _list.append(self._force_time(i))
        return _list

    def _force_time(self, x):
        t = Symbol('t')
        if self.force is not None:
            if self.omega and self.a is not None:
                f =  ((self.a) * ( self.force(self.omega * t) ) )
                f = lambdify(t, f)
                return f(x)
            else:
                f = self.force(t)
                f = lambdify(t, f) 
                return f(x)
        else:
            return 10

class mass_damp_spring(force_time):
    def __init__(self, no_mass, range_mass, no_spring, range_spring, no_damp, range_damp, **kwargs): 
        super().__init__(**kwargs)
        self.mass = self._item_time(no_mass, range_mass)
        self.spring = self._item_time(no_spring, range_spring)
        self.damp = self._item_time(no_damp, range_damp)

    def _item_time(self, _no, _range):
        _len = len(self.time_array)
        _time_to_last = [0] * _no
        _item = []
        _final_item_list = []
        k = 0
        for i in range(_no):
            _item.append(randint(_range[0], _range[1]))
        for i in range(_len):
            _time_to_last[randint(0, _len) % _no] += 1 
        for i in _time_to_last:
            for j in range(i):
                _final_item_list.append(_item[k]) 
            k += 1
        return _final_item_list

class ideal_actual(mass_damp_spring):
    def __init__(self, delta_mass, delta_spring, delta_damp, **kwargs):
        super().__init__(**kwargs)
        self.delta_mass = delta_mass  
        self.delta_spring = delta_spring 
        self.delta_damp = delta_damp  
        self.ideal_values_list = self.ideal_values() 
        self.ideal_acc_list = self.ideal_acc()

    def ideal_diff(self, state, t):
        #call-back function to the ideal_values
        dx1dt = state[1] 
        T = t * self.product 
        dx2dt =  (1/self.mass[T])*(self.force_list[T] - (self.damp[T]*state[1]) - (self.spring[T] * state[0]) )
        dxdt = [dx1dt, dx2dt ]
        return dxdt 

    def ideal_values(self):
        #function to solve the differential equations
        return odeint(self.ideal_diff, self.state, self.time_array)
    
    def ideal_acc(self):
        #function to calculate ideal acceleration at a time_step
        ideal_acc_list = []
        z = self.ideal_values_list
        for i in self.time_array:
            T = self.product * i
            acc = (1/self.mass[T])*(self.force_list[T] - (self.damp[T]*z[T, 1]) - (self.spring[T]*z[T,0]) )
            ideal_acc_list.append(acc)
        return ideal_acc_list

class graphs(mass_damp_spring):
    def __init__(self, delta_mass, delta_K, delta_B,**kwargs):
        super().__init__(**kwargs) 
   
        self.delta_mass = delta_mass
        self.delta_K = delta_K  
        self.delta_B = delta_B
        self.actual_mass = self.mass + delta_mass 
        self.actual_k = self.k + delta_K 
        self.actual_b = self.b + delta_B  
        self.force_list = [] 

    def force_time(self, x):
        #equation to calculate force at a time step from the input equation
        t = Symbol('t')
        if self.force is not None:
            if self.omega and self.a is not None:
                f =  ((self.a) * ( self.force(self.omega * t) ) )
                f = lambdify(t, f)
                return f(x)
            else:
                f = self.force(t)
                f = lambdify(t, f) 
                return f(x)
        else:
            return 10
    
    def diff_force_time(self, x):
        #equation to calculate the differentiation of force at a time step
        t = Symbol('t')
        if self.force is not None:
            if self.omega and self.a is not None:
                f = ( (self.a) * ( self.force(self.omega * t) ) )
                f_prime = lambdify(t, f.diff(t)) 
                return f_prime(x)
            else: 
                f = self.force(t)
                f_prime = lambdify(t, f.diff(t))
                return f_prime(x)
        else:
            return 10

    def ideal_diff(self, state, t):
        #call-back function to the ideal_values
        dx1dt = state[1] 
        dx2dt =  (1/self.mass)*(self.force_time(t) - (self.b*state[1]) - (self.k * state[0]) )
        dxdt = [dx1dt, dx2dt ]
        return dxdt

    def ideal_values(self):
        #function to solve the differential equations
        return odeint(self.ideal_diff, self.state_vec, self.t)
    
    def ideal_acc(self):
        #function to calculate ideal acceleration at a time_step
        ideal_acc_list = []
        z = self.ideal
        for i in range(len(self.t)):
            t = self.t[i] 
            acc = (1/self.mass) * ( self.force_time(t) - (self.b*z[i,1]) - (self.k * z[i,0]) ) 
            ideal_acc_list.append(acc)
        return ideal_acc_list

    def actual_diff(self, state, t):
        #callback function to calculate actual_values
        dx1dt = state[1] 
        dx2dt = (1/self.actual_mass)*(self.force_time(t) - (self.actual_b*state[1]) - (self.actual_k*state[0]) )
        dxdt = [dx1dt, dx2dt] 
        return dxdt

    def actual_values(self):
        #function to solve the differentiation equations and give actual displacement and velocity 
        return odeint(self.actual_diff, self.state_vec, self.t)
    
    def actual_acc(self):
        #function to calculate actual acceleration at a time_step
        actual_acc_list = []
        z = self.actual
        for i in range(len(self.t)):
            t = self.t[i] 
            acc = (1/self.actual_mass) * ( self.force_time(t) - (self.actual_b*z[i,1]) - (self.actual_k*z[i,0]) ) 
            actual_acc_list.append(acc)
        return actual_acc_list
    
    def error_equation(self): 
        # function to calculate G(x)
        z = self.actual
        z1 = list(z[:,0])
        z2 = list(z[:,1])
        z3 = self.actual_acc()
        error = []
        for i in range(len(self.t)):
            e = (self.delta_mass * z3[i]) + (self.delta_B * z2[i]) + (self.delta_K * z1[i])
            error.append(e) 
        return error 
    
    def force_graph(self):
        #plots the force_time graph
        for i in range(len(self.t)):
            self.force_list.append(self.force_time(self.t[i]))
        self.force_array = np.array(self.force_list)
        plt.plot(self.t, self.force_array)
        plt.xlabel('TIME')
        plt.ylabel('FORCE')
        plt.grid()
        plt.show()
    
    def ideal_graph(self):
        #plots the graph of ideal values
        z = self.ideal
        disp = z[:, 0]
        vel = z[:, 1]
        acc = self.ideal_acc() 
        plt.plot(self.t, disp)
        plt.plot(self.t, vel)
        plt.plot(self.t, acc)
        plt.title('Spring_damper_system')
        plt.xlabel("TIME")
        plt.ylabel('disp_vel_acc')
        plt.legend(["disp", "vel", "acc"])
        plt.grid()
        plt.show()
    
    def actual_graph(self):
        #plots the graph of actual values
        z = self.actual
        disp = z[:, 0]
        vel = z[:, 1] 
        acc = self.actual_acc()
        plt.plot(self.t, disp)
        plt.plot(self.t, vel)
        plt.plot(self.t, acc)
        plt.title('Actual_Spring_damper_system')
        plt.xlabel("TIME")
        plt.ylabel('actual_disp_vel_acc')
        plt.legend(["actual_disp", "actual_vel", "actual_acc"])
        plt.grid()
        plt.show()

    def compare_disp(self):
        #plots to compare actual and ideal displacement
        i = self.ideal
        a = self.actual
        actual_disp = a[:, 0]
        ideal_disp = i[:, 0] 
        plt.plot(self.t, actual_disp)
        plt.plot(self.t, ideal_disp)
        plt.title('actual_ideal_disp')
        plt.legend(['actual_disp', 'ideal_disp'])
        plt.grid()
        plt.show()

    def compare_vel(self):
        #plots to compare actual and ideal displacement
        i = self.ideal
        a = self.actual
        actual_vel = a[:, 1]
        ideal_vel = i[:, 1] 
        plt.plot(self.t, actual_vel)
        plt.plot(self.t, ideal_vel)
        plt.title('actual_ideal_velocity')
        plt.legend(['actual_vel', 'ideal_vel'])
        plt.grid()
        plt.show()
    
    def compare_acc(self):
        #plots to compare actual and ideal acceleration
        ideal_acc = self.ideal_acc()
        actual_acc = self.actual_acc()  
        plt.plot(self.t, actual_acc)
        plt.plot(self.t, ideal_acc) 
        plt.title('actual_ideal_acceleration')
        plt.legend(['actual_acc', 'ideal_acc'])
        plt.grid()
        plt.show()

    def actual_values_csv(self):
        #actual values dataframe
        actual_z = self.actual
        ideal_z = self.ideal
        ideal_acc = self.ideal_acc()
        actual_acc = self.actual_acc()
        error = self.error_equation() 
        D = {"time_step" : self.t,
                "Amplitude" : self.a,
                "Angular_Vel" : self.omega,
                "force" : self.force_array,
                "mass" : self.mass,
                "K" : self.k,
                "B" : self.b, 
                "delta_mass" : self.delta_mass,
                "delta_K" : self.delta_K,
                "delta_B" : self.delta_B,
                "ideal_disp" : ideal_z[:,0],
                "ideal_vel" : ideal_z[:,1],
                "ideal_acc" : ideal_acc,
                "actual_disp" : actual_z[:,0],
                "actual_vel" : actual_z[:,1],
                "actual_acc" : actual_acc,
                "G(x)" : error
                }

        df = pd.DataFrame(D)
        return df 
