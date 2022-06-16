"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch

from filing_paths import path_model

import sys
sys.path.insert(1, path_model)
sys.path.insert(1, "Simulations/Pendulum/model.py")

from model import getJacobian
#from EKF_pendulum import getJacobian

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

class ExtendedKalmanFilter:

    def __init__(self, SystemModel, mode='full'):
        self.f = SystemModel.f
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q

        self.h = SystemModel.h
        self.n = SystemModel.n

        # Has to be transformed because of EKF non-linearity
        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # Pre allocate KG array
        self.KG_array = torch.zeros((self.T_test, self.m, self.n))

        #Changes the Jacobian used for linearization
        if(mode == 'full'):
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'
        elif(mode == 'partial'):
            self.fString = 'ModAcc'
            self.hString = 'ObsPar'
        elif(mode == "nonlinear"):
            self.fString = "ModAcc"
            self.hString = "NonLinear"
   
    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior))
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(self.m1x_posterior, self.fString), getJacobian(self.m1x_prior, self.hString))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        #print(f"m1_x prior :{self.m1x_prior}")
        #print(f"sigma prediced :{self.m2x_prior}")
        #print(f"H_t:{self.H_T}")
        #print(f"m2x_prior:{self.m2x_prior}")
        #print(f"m2y:{self.m2y}")
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))
        #Save KalmanGain
        self.KG_array[self.i] = self.KG
        self.i += 1
        #print(f"kalman_gain :{self.KG}")
        #assert torch.all(self.KG >= 0)


    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

        #JOSEPH-STABILIZED FORM (MAY OR MAY NOT BE BETTER)
        """
        self.first_term = torch.eye(self.m) - torch.matmul(self.KG, self.H)
        self.m2x_posterior = torch.matmul(self.first_term, self.m2x_prior)
        self.m2x_posterior = torch.matmul(self.m2x_posterior, torch.transpose(self.first_term, 0, 1))
        self.last_term = torch.matmul(torch.matmul(self.KG, self.R), torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_posterior + self.last_term
        """

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    def UpdateJacobians(self, F, H):
        self.F = F
        self.F_T = torch.transpose(F, 0, 1)
        self.H = H
        self.H_T = torch.transpose(H, 0, 1)
        #print(self.H,self.F,'\n')
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T])
        self.sigma = torch.empty(size=[self.m, self.m, T])
        # Pre allocate KG array
        self.KG_array = torch.zeros((T, self.m, self.n))
        self.i = 0 # Index for KG_array alocation

        self.m1x_posterior = torch.squeeze(self.m1x_0)
        self.m2x_posterior = torch.squeeze(self.m2x_0)

        for t in range(0, T):
            yt = y[:, t]
            xt, sigmat = self.Update(yt)
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)