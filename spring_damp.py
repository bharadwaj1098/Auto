import numpy as np
from scipy.integrate import odeint
import pandas as pd
import random
from random import randint 
from sympy import *
import matplotlib.pyplot as plt
random.seed(49)

class force_time():
    def __init__(self, * ,time, increment,state,force= None, amplitude=None, omega=None):
        self.state=state
        self.increment = increment
        self.product = int(1 / (self.increment))
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
            if self.omega and self.amplitude is not None:
                f =  ((self.amplitude) * ( self.force(self.omega * t) ) )
                f = lambdify(t, f)
                return f(x)
            else:
                f = self.force(t)
                f = lambdify(t, f) 
                return f(x)
        else:
            return 10

class mass_damp_spring(force_time):
    def __init__(self, *,no_mass, range_mass, no_spring, range_spring, no_damp, range_damp, uniform=None, **kwargs): 
        super().__init__(**kwargs)
        #self.uniform = uniform
        self.mass = self._item_time(no_mass, range_mass, uniform=None)
        self.spring = self._item_time(no_spring, range_spring, uniform)
        self.damp = self._item_time(no_damp, range_damp, uniform)

    def _item_time(self, _no, _range, uniform):
        _len = len(self.time_array)
        _time_to_last = [0] * _no
        _item = []
        _final_item_list = []
        k = 0
        for i in range(_no):
            if uniform is not None: 
                _item.append(random.uniform(_range[0], _range[1]))
            else:
                _item.append(randint(_range[0], _range[1]))
        for i in range(_len):
            _time_to_last[randint(0, _len) % _no] += 1 
        for i in _time_to_last:
            for j in range(i):
                _final_item_list.append(_item[k]) 
            k += 1
        return _final_item_list

class ideal_actual(mass_damp_spring):
    def __init__(self, *,delta_mass, delta_spring, delta_damp, **kwargs):
        super().__init__(**kwargs)
        self.delta_mass = delta_mass  
        self.delta_spring = delta_spring 
        self.delta_damp = delta_damp  
        self.ideal_values_list = self.ideal_values() 
        self.ideal_acc_list = self.ideal_acc()
        self.actual_values_list = self.actual_values()
        self.actual_acc_list = self.actual_acc()
        self.error_list = self.error()

    def ideal_diff(self, state, t):
        #call-back function to the ideal_values
        dx1dt = state[1] 
        T = int(t * self.product )
        dx2dt =  (1/self.mass[T])*(self.force_list[T] - (self.damp[T]*state[1]) - (self.spring[T] * state[0]) )
        dxdt = [dx1dt, dx2dt ]
        return dxdt 
    
    def actual_diff(self, state, t):
        dx1dt = state[1]
        T = int(t* self.product) 
        m = 1/(self.mass[T] + self.delta_mass) 
        f = self.force_list[T] 
        b = (self.damp[T] + self.delta_damp)*state[1] 
        k = (self.spring[T] + self.delta_spring)*state[0]
        dx2dt = m * (f - b - k) 
        dxdt = [dx1dt, dx2dt] 
        return dxdt  

    def ideal_values(self):
        #function to solve the differential equations
        return odeint(self.ideal_diff, self.state, self.time_array)
    
    def actual_values(self):
        return odeint(self.actual_diff, self.state, self.time_array)

    def ideal_acc(self):
        #function to calculate ideal acceleration at a time_step
        ideal_acc_list = []
        z = self.ideal_values_list
        for i in self.time_array:
            T = int(self.product * i)
            acc = (1/self.mass[T])*(self.force_list[T] - (self.damp[T]*z[T, 1]) - (self.spring[T]*z[T,0]) )
            ideal_acc_list.append(acc)
        return ideal_acc_list
    
    def actual_acc(self): 
        actual_acc_list = []
        z = self.actual_values_list
        for i in self.time_array:
            T = int(self.product * i)
            m = 1/(self.mass[T] + self.delta_mass) 
            f = self.force_list[T] 
            b = (self.damp[T] + self.delta_damp)*z[T,1] 
            k = (self.spring[T] + self.delta_spring)*z[T,0]
            acc = m*(f - b - k )
            actual_acc_list.append(acc)
        return actual_acc_list

    def error(self):
        # function to calculate G(x)
        z = self.actual_values_list
        z1 = list(z[:,0])
        z2 = list(z[:,1])
        z3 = self.actual_acc()
        error = []
        for i in range(len(self.time_array)):
            e = (self.delta_mass * z3[i]) + (self.delta_damp * z2[i]) + (self.delta_spring * z1[i])
            error.append(e) 
        return error

class graphs(ideal_actual):
    def __init__(self, *, k = None, **kwargs):
        super().__init__(**kwargs)
    
    def force_graph(self):
        plt.plot(self.time_array, self.force_list)
        plt.xlabel('TIME')
        plt.ylabel('FORCE')
        plt.title('FORCE_GRAPH')
        plt.grid()
        plt.show()

    def g_x(self):
        plt.plot(self.time_array, self.error_list)
        plt.xlabel('TIME')
        plt.ylabel('G(X)')
        plt.title('G(X)_GRAPH')
        plt.grid()
        plt.show()

    def mass_graph(self):
        plt.plot(self.time_array, self.mass)
        plt.xlabel('TIME')
        plt.ylabel('MASS')
        plt.title('MASS_Graph')
        plt.grid()
        plt.show()

    def spring_graph(self):
        plt.plot(self.time_array, self.spring)
        plt.xlabel('TIME')
        plt.ylabel('SPRING') 
        plt.title('SPRING_GRAPH')
        plt.grid()
        plt.show()

    def damp_graph(self):
        plt.plot(self.time_array, self.damp)
        plt.xlabel('TIME')
        plt.ylabel('DAMP') 
        plt.title('DAMP_GRAPH')
        plt.grid()
        plt.show()

    def ideal_graph(self):
        #plots the graph of ideal values
        plt.plot(self.time_array, self.ideal_values_list[:, 0])
        plt.plot(self.time_array, self.ideal_values_list[:, 1])
        plt.plot(self.time_array, self.ideal_acc_list)
        plt.title('Spring_damper_system')
        plt.xlabel("TIME")
        plt.ylabel('disp_vel_acc')
        plt.legend(["disp", "vel", "acc"])
        plt.grid()
        plt.show()
    
    def actual_graph(self):
        #plots the graph of actual values 
        plt.plot(self.time_array, self.actual_values_list[:, 0])
        plt.plot(self.time_array, self.actual_values_list[:, 1])
        plt.plot(self.time_array, self.actual_acc_list)
        plt.title('Actual_Spring_damper_system')
        plt.xlabel("TIME")
        plt.ylabel('actual_disp_vel_acc')
        plt.legend(["actual_disp", "actual_vel", "actual_acc"])
        plt.grid()
        plt.show()

    def compare_disp(self):
        #plots to compare actual and ideal displacement 
        plt.plot(self.time_array, self.actual_values_list[:, 0])
        plt.plot(self.time_array, self.ideal_values_list[:, 0])
        plt.title('actual_ideal_disp')
        plt.legend(['actual_disp', 'ideal_disp'])
        plt.grid()
        plt.show()

    def compare_vel(self):
        #plots to compare actual and ideal displacement
        plt.plot(self.time_array, self.actual_values_list[:, 1])
        plt.plot(self.time_array, self.ideal_values_list[:, 1])
        plt.title('actual_ideal_velocity')
        plt.legend(['actual_vel', 'ideal_vel'])
        plt.grid()
        plt.show()
    
    def compare_acc(self):
        #plots to compare actual and ideal acceleration 
        plt.plot(self.time_array, self.actual_acc_list)
        plt.plot(self.time_array, self.ideal_acc_list) 
        plt.title('actual_ideal_acceleration')
        plt.legend(['actual_acc', 'ideal_acc'])
        plt.grid()
        plt.show()

    def actual_values_csv(self):
        #actual values dataframe
        D = {"time_step" : self.time_array,
                "Amplitude" : self.amplitude,
                "Angular_Vel" : self.omega,
                "force" : self.force_list,
                "mass" : self.mass,
                "K" : self.spring,
                "B" : self.damp, 
                "delta_mass" : self.delta_mass,
                "delta_K" : self.delta_spring,
                "delta_B" : self.delta_damp,
                "ideal_disp" : self.ideal_values_list[:,0],
                "ideal_vel" : self.ideal_values_list[:,1],
                "ideal_acc" : self.ideal_acc_list,
                "actual_disp" : self.actual_values_list[:,0],
                "actual_vel" : self.actual_values_list[:,1],
                "actual_acc" : self.actual_acc_list,
                "G(x)" : self.error_list
                }

        df = pd.DataFrame(D)
        return df 
