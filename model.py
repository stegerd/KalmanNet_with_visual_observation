import math
import torch
import numpy as np
# torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd
#from parameters import NL_m, NL_n, H_design

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
#from PendulumData import Pendulum

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")



delta_t = 5*1e-3
delta_t_gen = 1e-5
variance = 0


def f(x):
    g = 9.81
    l = 1
    return torch.squeeze(torch.Tensor([(x[0] + x[1]*delta_t), (x[1] - g/l * torch.sin(x[0]) * delta_t)]))

def h_full(x):
    H = torch.eye(2)
    return torch.matmul(H, x)


def h_partial(x):
    H = torch.Tensor([[1, 0]])
    return torch.matmul(H, x)

def h_nonlinear(x):
    return torch.Tensor([torch.sin(x[0])])



def getJacobian(x, a):
    g=9.81
    l=1
    if a == 'ModAcc':
        return torch.Tensor([[1, delta_t], [(-g/l * torch.cos(x[0]) * delta_t), 1]])
    if a == 'ObsAcc':
        return torch.eye(2)
    if a == "ObsPar":
        return torch.Tensor([[1, 0]])
    if a == "NonLinear":
        return torch.Tensor([[torch.cos(x[0]), 0]])






"""

img_size = 24

pend_params = Pendulum.pendulum_default_params()
pend_params[Pendulum.FRICTION_KEY] = 0.1

data = Pendulum(img_size=img_size,
                observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                transition_noise_std=0.001,
                observation_noise_std=1e-5,
                pendulum_params=pend_params,
                seed=0)


def f_gen(x):
    g = 9.81 # Gravitational Acceleration
    L = 1 # Radius of pendulum
    result = [x[0]+x[1]*delta_t_gen, x[1]-(g/L * torch.sin(x[0]))*delta_t_gen]
    result = torch.squeeze(torch.tensor(result))
    # print(result.size())
    return result

# def f(x):
#     g = 9.81 # Gravitational Acceleration
#     L = 1 # Radius of pendulum
#     result = [x[0]+x[1]*delta_t, x[1]-(g/L * torch.sin(x[0]))*delta_t]
#     result = torch.squeeze(torch.tensor(result))
#     # print(result.size())
#     return result

def f(x):
    x_np = x.detach().cpu().numpy()
    x = data._transition_function_KNet(x_np)
    return x


def h(x):
    #return torch.matmul(H_design,x).to(dev)
    return torch.matmul(H_design.to(dev, non_blocking=True), x)
    #return toSpherical(x)

def fInacc(x):
    g = 9.81 # Gravitational Acceleration
    L = 1.1 # Radius of pendulum
    result = [x[0]+x[1]*delta_t, x[1]-(g/L * torch.sin(x[0]))*delta_t]
    result = torch.squeeze(torch.tensor(result))
    # print(result.size())
    return result

# def hInacc(x):
#     return torch.matmul(H_mod,x)
#     #return toSpherical(x)

def getJacobian(x, a):
    
    # if(x.size()[1] == 1):
    #     y = torch.reshape((x.T),[x.size()[0]])
    try:
        if(x.size()[1] == 1):
            y = torch.reshape((x.T),[x.size()[0]])
    except:
        y = torch.reshape((x.T),[x.size()[0]])
        
    if(a == 'ObsAcc'):
        g = h
    elif(a == 'ModAcc'):
        g = f
    elif(a == 'ObsInacc'):
        g = hInacc
    elif(a == 'ModInacc'):
        g = fInacc

    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1,NL_m)
    return Jac



# def hInv(y):
#     return torch.matmul(H_design_inv,y)
#     #return toCartesian(y)


# def hInaccInv(y):
#     return torch.matmul(H_mod_inv,y)
#     #return toCartesian(y)

'''
x = torch.tensor([[1],[1],[1]]).float() 
H = getJacobian(x, 'ObsAcc')
print(H)
print(h(x))

F = getJacobian(x, 'ModAcc')
print(F)
print(f(x))
'''
"""