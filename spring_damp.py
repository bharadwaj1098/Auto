import numpy as np
from scipy.integrate import odeint
import pandas as pd
from sympy import *
import matplotlib.pyplot as plt

class spring_damp_mass():
    
    def __init__(self, state_vec, time_in_sec, mass,
                 K, B, delta_mass, delta_K, delta_B,
                 amplitude= None, omega= None,force = None):
        self.k = K 
        self.b = B
        self.mass = mass
        self.actual_mass = self.mass + delta_mass 
        self.actual_k = self.k + delta_K 
        self.actual_b = self.b + delta_B  
        self.state_vec = state_vec
        self.force_list = []
        self.force = force
        self.a = amplitude
        self.omega = omega
        self.t = np.arange(0, time_in_sec+0.1, 0.1)  

    def force_time(self, x):
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
        dx1dt = state[1] 
        dx2dt = state[2]
        #(1/self.mass)*( self.force_time(t) - (self.b*state[1]) - (self.k * state[0]) )
        dx3dt = (1/self.mass)*(self.diff_force_time(t) - (self.b*state[2]) - (self.k * state[1]) )
        dxdt = [dx1dt, dx2dt, dx3dt] 
        return dxdt

    def ideal_values(self):
        return odeint(self.ideal_diff, self.state_vec, self.t)
    
    def actual_diff(self, state, t):
        dx1dt = state[1] 
        dx2dt = state[2]
        #(1/self.mass)*( self.force_time(t) - (self.b*state[1]) - (self.k * state[0]) )
        dx3dt = (1/self.actual_mass)*(self.diff_force_time(t) - (self.actual_b*state[2]) - (self.actual_k * state[1]) )
        dxdt = [dx1dt, dx2dt, dx3dt] 
        return dxdt

    def actual_values(self):
        return odeint(self.actual_diff, self.state_vec, self.t)
        
    def force_graph(self):
        for i in range(len(self.t)):
            self.force_list.append(self.force_time(self.t[i]))
        self.force_array = np.array(self.force_list)
        plt.plot(self.t, self.force_array)
        plt.xlabel('TIME')
        plt.ylabel('FORCE')
        plt.grid()
        plt.show()
    
    def ideal_graph(self):
        z = self.ideal_values()
        disp = z[:, 0]
        vel = z[:, 1]
        acc = z[:, 2]
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
        z = self.actual_values()
        disp = z[:, 0]
        vel = z[:, 1]
        acc = z[:, 2]
        plt.plot(self.t, disp)
        plt.plot(self.t, vel)
        plt.plot(self.t, acc)
        plt.title('Spring_damper_system')
        plt.xlabel("TIME")
        plt.ylabel('actual_disp_vel_acc')
        plt.legend(["actual_disp", "actual_vel", "actual_acc"])
        plt.grid()
        plt.show()
    
    def actual_values_csv(self):
        z = self.actual_values() 
        #columns = ['time_step', 'force', 'mass', 'K', 'B', 'actual_disp', 'actual_vel', 'actual_acc']
        D = {"time_step" : self.t,
                "force" : self.force_array,
                "mass" : self.mass,
                "K" : self.k,
                "B" : self.b,
                "actual_disp" : z[:,0],
                "actual_vel" : z[:,1],
                "actual_acc" : z[:,2]}
        df = pd.DataFrame(D, index=None)
        return df
