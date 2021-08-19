import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sympy import * 
import random
import yaml
import sys
from scipy.integrate import odeint  

from haptic import haptic_class, haptic_actual
from spring_damp import mass_damp_spring 
from random import randint 

number = 5 #number of different dynamics +

# m=0.005-0.01, b=0.1-0.2, k=1-10
if __name__ == '__main__':
    name = sys.argv[1]
    with open(name) as File:
        d = yaml.load(File, Loader=yaml.FullLoader)
    
    flag = d['non_linear']

    if flag != 'yes':

        for i in range(d['number_of_runs']): #lets see how many we can manage to get

            m1 = mass_damp_spring(time = d['time'], increment = d['increment'], state= d['state'], #state_actually doesn't matter for sprring_damp_class
                            force = sin, amplitude= d['args_dict_h']['amplitude'], omega=d['args_dict_h']['omega'], 
                            no_mass = d['args_dict_h']['no_mass'], range_mass= d['args_dict_h']['range_mass'], 
                            no_spring= d['args_dict_h']['no_spring'], range_spring = d['args_dict_h']['range_spring'], 
                            no_damp = d['args_dict_h']['no_damp'], range_damp= d['args_dict_h']['range_damp'], uniform=1) 

            m2 = mass_damp_spring(time = d['time'], increment = d['increment'], state= d['state'], #state_actually doesn't matter for sprring_damp_class
                            force = cos, amplitude= d['args_dict_h']['amplitude'], omega=d['args_dict_h']['omega'], 
                            no_mass = d['args_dict_h']['no_mass'], range_mass= d['args_dict_h']['range_mass'], 
                            no_spring= d['args_dict_h']['no_spring'], range_spring = d['args_dict_h']['range_spring'], 
                            no_damp = d['args_dict_h']['no_damp'], range_damp= d['args_dict_h']['range_damp'], uniform=1)
            
            print(i, len(m1.mass), len(m2.mass), f'non_linear :{flag}') 

            hap = haptic_class(dic = d, Theta_h=sin, Theta_a=cos) 
            df = hap.dataframe()
            
            if d['test'] == 'no':
                path = f'Data/Linear_error/linear_error_{i+7}.csv'  #if code breaks inbetween change {i+1} to number of files in {Linear_error folder + i}
                                                                    # in normal case keep it {i+1}
            
            elif d['test'] == 'yes':
                path = f'Data/Linear_error/Test_data_{i+1}.csv'

            df.to_csv(path)

    elif flag == 'yes':

        for i in range(d['number_of_runs'] ):
            m1 = mass_damp_spring(time = d['time'], increment = d['increment'], state= d['state'], #state_actually doesn't matter for sprring_damp_class
                            force = sin, amplitude= d['args_dict_h']['amplitude'], omega=d['args_dict_h']['omega'], 
                            no_mass = d['args_dict_h']['no_mass'], range_mass= d['args_dict_h']['range_mass'], 
                            no_spring= d['args_dict_h']['no_spring'], range_spring = d['args_dict_h']['range_spring'], 
                            no_damp = d['args_dict_h']['no_damp'], range_damp= d['args_dict_h']['range_damp'], uniform=1) 

            m2 = mass_damp_spring(time = d['time'], increment = d['increment'], state= d['state'], #state_actually doesn't matter for sprring_damp_class
                            force = cos, amplitude= d['args_dict_h']['amplitude'], omega=d['args_dict_h']['omega'], 
                            no_mass = d['args_dict_h']['no_mass'], range_mass= d['args_dict_h']['range_mass'], 
                            no_spring= d['args_dict_h']['no_spring'], range_spring = d['args_dict_h']['range_spring'], 
                            no_damp = d['args_dict_h']['no_damp'], range_damp= d['args_dict_h']['range_damp'], uniform=1)
            
            print(i, len(m1.mass), len(m2.mass), f'linear_or_not : {flag}') 

            hap = haptic_class(dic = d, Theta_h=sin, Theta_a=cos) 
            df = hap.dataframe()
            
            if d['test'] == 'no':
                path = f'Data/Non_Linear/non_linear_error_{i+4}.csv'  #if code breaks inbetween change {i+1} to number of files in {Linear_error folder + i}
                                                                    # in normal case keep it {i+1}
            
            elif d['test'] == 'yes':
                path = f'Data/Non_Linear/Non_Test_data_{i+1}.csv'

            df.to_csv(path)