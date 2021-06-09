import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
torch.manual_seed(10)

import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings("ignore") 
import time 
import random 
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline 
import matplotlib.pyplot as plt

#from spring_damp import spring_damp_mass


if __name__ == "__main__":
    df = pd.read_csv('sim_1.csv') 

    imp_colums = ["time_step", "force", "mass", "K", "B","actual_disp", "actual_vel", "actual_acc", "G(x)"] 

    for i in df.columns:
        if i not in imp_colums:
            df = df.drop([i], axis=1)
    
    output = df["G(x)"]
    df = df.drop(["G(x)"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df, output, random_state=42)

    pipe = make_pipeline(StandardScaler(), LinearRegression())
    pipe_2 = make_pipeline(StandardScaler()) 
    
    pipe.fit(X_train, y_train)
    z = pipe.score(X_test, y_test)
    x = pipe_2.fit(y_test)
    y = pipe.predict(X_test)
    
    plt.plot(df['time_step'], x)
    plt.plot(df['time_step'], y) 
    plt.legend(['real', 'prediction'])

