import sys
import os
'''
parent_dir='C:\\Users\\garav\\AGOP\\DLR\\utils'
model_dir= 'C:\\Users\\garav\\AGOP\\DLR\\trained_models\\MNIST\\model4\\nn_models\\'
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
'''
#parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#print("parent",parent_dir)
parent_dir= os.path.join('/work/DLR','utils/')
sys.path.append(parent_dir)