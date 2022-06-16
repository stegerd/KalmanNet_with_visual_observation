import torch.nn as nn
import torch
import time
from EKF_visual import ExtendedKalmanFilter
import numpy as np

def EKFTest(SysModel, test_input, test_target, model_AE_conv, matrix_data_flag, modelKnowledge = 'full', allStates=True, true_init_flag=0, random_init_flag=0, boundary=np.pi/2):
    """
    Takes noisy sequence of observations, and ground truth sequence, a model class and possibly and encoder. Returns the EKF
    prediction of each trajectory.
    :param SysModel: SystemModel object, contains definition of f, h, Q, R, number of trajectories, length of trajectory
    as well as potential init point of trajectories
    :param test_input: observations
    :param test_target: ground truth state
    :param model_AE_conv: Trained encoder which takes as input the visual observation (image)
    :param matrix_data_flag: True if observations are given in vectorial form, false if obs. in image form
    :param modelKnowledge: string needed for EKF class, "full" if H is eye(n), "partial" if H is eye(n)-eye(k), k < n;
    "nonlinear" is H is a nonlinear function of x.
    :param allStates:
    :param true_init_flag: True if m_1x0 and m2x_0 are known and taken from ground truth sequence, False if else
    :param random_init_flag: True if m_1x0, m2x_0 are set randomly, sampled from uniform distribution
    :param boundary: used when random_init_flag=1, is the +- boundary of the uniform distribtuion
    :return: array: avg total MSE of each trajectory in linear scale, avg theta marginal MSE of each trajectory in linear scale,
    avg omega marginal MSE in linear scale, avg total MSE of each trajectory in log scale, avg theta marginal MSE of each trajectory in log scale,
    avg omega marginal MSE in log scale, std of total MSE per traj. log scale, std of theta marginal MSE per traj. log scale,
    std of omega marginal MSE per traj. log scale, sequence of KG matrix averaged over trajectories , full predicted trajectories
    """
    #number of sequences
    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    
    # MSE per sequence, total: MSE over all states, theta: MSE on position only, omega: MSE on velocity only
    MSE_EKF_linear_arr_total = torch.empty(N_T)
    MSE_EKF_linear_arr_theta = torch.empty(N_T)
    MSE_EKF_linear_arr_omega = torch.empty(N_T)

    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    if (not random_init_flag) and (not true_init_flag):
        EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(EKF.KG_array)
    y_test_decoaded = torch.empty([N_T, SysModel.n, SysModel.T_test])
    EKF_out = torch.empty([N_T, SysModel.m, SysModel.T_test])
    start = time.time()

    for j in range(0, N_T):
        if matrix_data_flag:
            y_test_decoaded = test_input
        else: # use the output of trained encoder as the input of KF
            y_mdl_tst = test_input[j:j+1, ...]
            for t in range(0, SysModel.T_test):
                AE_input = y_mdl_tst[:, t+1:t+2, :, :]
                model_AE_conv.eval()
                with torch.no_grad():
                    y_test_decoaded[j, :, t] = model_AE_conv(AE_input)
        if random_init_flag:
            m_0 = np.random.uniform(low=-boundary, high=+boundary)
            EKF.InitSequence(torch.Tensor([m_0, 0]), SysModel.m2x_0)
        if true_init_flag:
            m_0 = test_target[j, 0, 0]
            EKF.InitSequence(torch.Tensor([m_0, 0]), SysModel.m2x_0)
        if matrix_data_flag:
            EKF.GenerateSequence(y_test_decoaded[j, :, 1:], EKF.T_test)
        else:
            EKF.GenerateSequence(y_test_decoaded[j, :, :], EKF.T_test)

        if(allStates):
            MSE_EKF_linear_arr_total[j] = loss_fn(EKF.x, test_target[j, :, 1:]).item()
            MSE_EKF_linear_arr_theta[j] = loss_fn(EKF.x[0], test_target[j, 0, 1:]).item()
            MSE_EKF_linear_arr_omega[j] = loss_fn(EKF.x[1], test_target[j, 1, 1:]).item()

        else:
            loc = torch.tensor([True,False,True,False])
            MSE_EKF_linear_arr_total[j] = loss_fn(EKF.x[loc,:], test_target[j, :, :]).item()

        KG_array = torch.add(EKF.KG_array, KG_array) 
        EKF_out[j, :, :] = EKF.x

    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    KG_array /= N_T

    MSE_EKF_linear_avg_total = torch.mean(MSE_EKF_linear_arr_total)
    MSE_EKF_dB_avg_total = 10 * torch.log10(MSE_EKF_linear_avg_total)

    MSE_EKF_linear_avg_theta = torch.mean(MSE_EKF_linear_arr_theta)
    MSE_EKF_dB_avg_theta = 10 * torch.log10(MSE_EKF_linear_avg_theta)

    MSE_EKF_linear_avg_omega = torch.mean(MSE_EKF_linear_arr_omega)
    MSE_EKF_dB_avg_omega = 10 * torch.log10(MSE_EKF_linear_avg_omega)

    
    # Standard deviation
    MSE_EKF_std_total = torch.std(MSE_EKF_linear_arr_total, unbiased=True)
    MSE_EKF_std_omega = torch.std(MSE_EKF_linear_arr_omega, unbiased=True)
    MSE_EKF_std_theta = torch.std(MSE_EKF_linear_arr_theta, unbiased=True)

    MSE_EKF_dB_std_total = 10* torch.log10(MSE_EKF_linear_avg_total + MSE_EKF_std_total) - MSE_EKF_dB_avg_total
    MSE_EKF_dB_std_theta = 10* torch.log10(MSE_EKF_linear_avg_theta + MSE_EKF_std_theta) - MSE_EKF_dB_avg_theta
    MSE_EKF_dB_std_omega = 10* torch.log10(MSE_EKF_linear_avg_omega + MSE_EKF_std_omega) - MSE_EKF_dB_avg_omega
    
    print("EKF - MSE LOSS TOTAL:", MSE_EKF_dB_avg_total, "[dB]")
    print("EKF - MSE STD TOTAL:", MSE_EKF_dB_std_total, "[dB]")
    print("Inference Time:", t)    # Print Run Time


    return [MSE_EKF_linear_arr_total, MSE_EKF_linear_arr_theta, MSE_EKF_linear_arr_omega, MSE_EKF_dB_avg_total, MSE_EKF_dB_avg_theta,
            MSE_EKF_dB_avg_omega, MSE_EKF_dB_std_total, MSE_EKF_dB_std_theta, MSE_EKF_dB_std_omega, KG_array, EKF_out]



