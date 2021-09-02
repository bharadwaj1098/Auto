# Auto

Implenting the Equations of motion for Haptic shared control from the paper "Modulation of Control Authority in Adaptive HapticShared Control Paradigms"  (https://arxiv.org/abs/2007.07436 )

## Repository discription

* Spring_damp.py and haptic.py hold all the classes to generate the data and simple examples of how to use and save them can be seen in the Notebooks->haptic_graph.ipynb.
* data_generation.py can be used to make several datasets. It can be run by the command “python3 data_generation.py Human_Auto.yaml”
* Human_Auto.yaml has all the inputs to the system to generate data. 
* Models directory consists of 3 folders ANN, BNN, and DeepGp. And 3 jupyter notebooks.
* The ANN and BNN folder consist of saved models for artificial neural networks and bayesian neural networks respectively.
* Deep_gp has 3 folders Linear_Multiclass, Non_Linear_Multiclass, Non_Linear_Uniclass.
* The 3 jupyter notebooks, in the Deep_gp directory, have been used to train the models. The example of loading and testing the model can be seen there. (NOTE:- don’t run the training part in the above-mentioned notebooks)