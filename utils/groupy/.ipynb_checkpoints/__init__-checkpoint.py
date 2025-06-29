import sys
import os
current_dir = os.getcwd()
#print(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
#print(parent_dir)
#model_dir = os.path.join(parent_dir, 'trained_models', 'CIFAR', 'model2', 'nn_models/')
#print(model_dir)
sys.path.append(parent_dir)

'''
import sys
parent_dir='C:\\Users\\garav\\AGOP\\DLR\\utils'
model_dir= 'C:\\Users\\garav\\AGOP\\DLR\\trained_models\\MNIST\\model4\\nn_models\\'
#parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
'''