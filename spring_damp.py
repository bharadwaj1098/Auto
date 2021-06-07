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
        self.delta_mass = delta_mass
        self.delta_K = delta_K  
        self.delta_B = delta_B
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
        dx2dt =  (1/self.mass)*(self.force_time(t) - (self.b*state[1]) - (self.k * state[0]) )
        dxdt = [dx1dt, dx2dt ]
        return dxdt

    def ideal_values(self):
        return odeint(self.ideal_diff, self.state_vec, self.t)
    
    def ideal_acc(self):
        ideal_acc_list = []
        z = self.ideal_values()
        for i in range(len(self.t)):
            t = self.t[i] 
            acc = (1/self.mass) * ( self.force_time(t) - (self.b*z[i,1]) - (self.k * z[i,0]) ) 
            ideal_acc_list.append(acc)
        return ideal_acc_list

    def actual_diff(self, state, t):
        dx1dt = state[1] 
        dx2dt = (1/self.actual_mass)*(self.force_time(t) - (self.actual_b*state[1]) - (self.actual_k*state[0]) )
        dxdt = [dx1dt, dx2dt] 
        return dxdt

    def actual_values(self):
        return odeint(self.actual_diff, self.state_vec, self.t)
    
    def actual_acc(self):
        actual_acc_list = []
        z = self.actual_values()
        for i in range(len(self.t)):
            t = self.t[i] 
            acc = (1/self.actual_mass) * ( self.force_time(t) - (self.actual_b*z[i,1]) - (self.actual_k*z[i,0]) ) 
            actual_acc_list.append(acc)
        return actual_acc_list
    
    def error_equation(self): 
        z = self.actual_values()
        z1 = list(z[:,0])
        z2 = list(z[:,1])
        z3 = self.actual_acc()
        error = []
        for i in range(len(self.t)):
            e = (self.delta_mass * z3[i]) + (self.delta_B * z2[i]) + (self.delta_K * z1[i])
            error.append(e) 
        return error 
    
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
        z = self.actual_values()
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
    
    def actual_values_csv(self):
        actual_z = self.actual_values()
        ideal_z = self.ideal_values()
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
        