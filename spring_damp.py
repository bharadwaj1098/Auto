import numpy as np
from scipy.integrate import odeint
import math
import matplotlib.pyplot as plt

class spring_damp_mass():
    def __init__(self, state_vec, time_in_sec, mass,
                 K, B, amplitude= None, omega= None,force = None):
        self.k = K 
        self.b = B 
        self.state_vec = state_vec
        self.force_list = []
        self.force = force
        self.a = amplitude
        self.omega = omega
        self.t = np.arange(0, time_in_sec+0.1, 0.1) 
        self.mass = mass 

    def force_time(self, t):
        if self.force is not None:
            if self.omega and self.a is not None:
                return (self.a) * ( self.force(self.omega * t) )
            else: 
                return self.force(t)
        else:
            return 10
    
    def diff(self, state, t):
        dx1dt = state[1] 
        dx2dt = (1/self.mass)*( self.force_time(t) - (self.b*state[1]) - (self.k * state[0]) )
        #dx3dt = (-1 * 1/self.mass) * ( (self.b * state[2] ) + (self.k * state[1]) ) 
        #dxdt = [dx1dt, dx2dt, dx3dt]
        dxdt = [dx1dt, dx2dt] 
        return dxdt

    def values(self):
        return odeint(self.diff, self.state_vec, self.t)
        
    def force_graph(self):
        for i in range(len(self.t)):
            self.force_list.append(self.force_time(self.t[i]))
        force_array = np.array(self.force_list)
        plt.plot(self.t, force_array)
        plt.xlabel('TIME')
        plt.ylabel('FORCE')
        plt.grid()
        plt.show()
    
    def disp_vel_graph(self):
        z = self.values()
        disp = z[:,0]
        vel = z[:,1]
        #acc = z[:, 2]
        plt.plot(self.t, disp)
        plt.plot(self.t, vel)
        #plt.plot(self.t, acc)
        plt.title('Spring_damper_system')
        plt.xlabel("TIME")
        plt.ylabel('disp_vel')
        plt.legend(["disp", "vel"])
        #plt.legend(["disp", "vel", "acc"])
        plt.grid()
        plt.show()
