import os

import torch
import sys
from datetime import datetime
from Extended_sysmdl_visual import SystemModel
import numpy as np
from EKF_test_visual import EKFTest

sys.path.insert(1, "Simulations/Pendulum/model.py")
from model import f, h_full, h_partial, h_nonlinear





if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        dev = torch.device("cpu")
        print("Running on the CPU")
    h = h_nonlinear
    NL_m = 2  # DIMENSION OF STATE
    NL_n = 1  # DIMENSION of observation
    NL_m1_0 = torch.FloatTensor([0.5, 0])

    NL_m2_0 = torch.tensor([[np.pi ** 2 / 3, 0.0], [0.0, 0.0]])
    NL_T = 400  # sequence length for training/cv set, here we not used
    NL_T_test = 400  # sequence length for test set
    modelKnow = "nonlinear"
    v = -10  # in [dB]
    q2s = [10, 2, 1, 0.5, 0.1, 0.01, 0.001, 0.0001]
    r2s = list(map(lambda x: x * (10 ** (-v / 10)), q2s))
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        dev = torch.device("cpu")
        print("Running on the CPU")
    for r2, q2 in zip(r2s, q2s):
        print(f"Evaluation for r2 {r2}, q2 {q2} \n")
        q2_grid = [q2, 1.1 * q2, 1.2 * q2, 1.3 * q2, 0.9 * q2, 0.8 * q2, 0.7 * q2]
        # q2_grid = [q2]
        best_q2 = q2
        best_MSE = 1000
        # input_ = np.load(f"/content/drive/MyDrive/ETH/Master thesis/KalmanNet_VO/Datasets/Pendulum/decimated_noisy_data/pendulum_decimated_noisy_q2_{q2}_r2_{r2}_v{v}.npz")
        # target = np.load(f"/content/drive/MyDrive/ETH/Master thesis/KalmanNet_VO/Datasets/Pendulum/decimated_clean_data/pendulum_decimated_q2_{q2}.npz")
        input = np.load(f"Datasets/Pendulum/decimated_clean_data/pendulum_decimated_q2_{q2}.npz")

        r = np.sqrt(r2)

        # INPUT MODIFICATION FOR NONLINEAR CASE, WE HAVE TO ADD OBSERVATION NOISE AFTER NONLINEAR TRANSFORMATION

        input_test = input["test_set"]
        target_test = input_test.copy()
        target_test = torch.from_numpy(target_test).float().to(dev)

        input_test = input_test[:, 0:1, :]
        input_test = np.sin(input_test) + np.random.randn(*input_test.shape) * r
        input_test = torch.from_numpy(input_test).float().to(dev)

        input_val = input["validation_set"][199:200, ...]
        target_val = input_val.copy()
        target_val = torch.from_numpy(target_val).float().to(dev)

        input_val = input_val[:, 0:1, :]
        input_val = np.sin(input_val) + np.random.randn(*input_val.shape) * r
        input_val = torch.from_numpy(input_val).float().to(dev)

        """
        input_test = torch.from_numpy(input_["test_set"][:, 0:1, :]).float().to(dev)
        target_test = torch.from_numpy(target["test_set"]).float().to(dev)
        input_val = torch.from_numpy(input_["validation_set"][:200, 0:1, :]).float().to(dev)
        target_val = torch.from_numpy(target["validation_set"][:200, ...]).float().to(dev)
        """
        print(f"Start grid search on validation set")
        for q2_g in q2_grid:
            print(f"Evaluate grid-value q2: {q2_g} ")
            q = np.sqrt(q2_g)
            sys_model = SystemModel(f, q, h, r, NL_T, NL_T_test, NL_m, NL_n, "pendulum")
            sys_model.InitSequence(NL_m1_0, NL_m2_0)

            [MSE_EKF_linear_arr_total, MSE_EKF_linear_arr_theta, MSE_EKF_linear_arr_omega, MSE_EKF_dB_avg_total,
             MSE_EKF_dB_avg_theta,
             MSE_EKF_dB_avg_omega, MSE_EKF_dB_std_total, MSE_EKF_dB_std_theta, MSE_EKF_dB_std_omega,
             KG_array, EKF_out] = EKFTest(SysModel=sys_model, test_input=input_val,
                                          test_target=target_val, model_AE_conv=None,
                                          matrix_data_flag=True, modelKnowledge=modelKnow)

            print(f"best_mSE: {best_MSE}, MSE: {MSE_EKF_dB_avg_total}")
            if MSE_EKF_dB_avg_total < best_MSE:
                best_MSE = MSE_EKF_dB_avg_total.item()
                best_q2 = q2_g

            print("\n")

        print(f"Evaluation on test set q_2: {q2}, with q2_best: {best_q2}")

        sys_model = SystemModel(f, np.sqrt(best_q2), h, r, NL_T, NL_T_test, NL_m, NL_n, "pendulum")
        sys_model.InitSequence(NL_m1_0, NL_m2_0)

        [MSE_EKF_linear_arr_total, MSE_EKF_linear_arr_theta, MSE_EKF_linear_arr_omega, MSE_EKF_dB_avg_total,
         MSE_EKF_dB_avg_theta,
         MSE_EKF_dB_avg_omega, MSE_EKF_dB_std_total, MSE_EKF_dB_std_theta, MSE_EKF_dB_std_omega,
         KG_array, EKF_out] = EKFTest(SysModel=sys_model, test_input=input_test, test_target=target_test,
                                      model_AE_conv=None, matrix_data_flag=True, modelKnowledge=modelKnow)
        sys_model = SystemModel(f, np.sqrt(q2), h, r, NL_T, NL_T_test, NL_m, NL_n, "pendulum")
        sys_model.InitSequence(NL_m1_0, NL_m2_0)

        [_, _, _, MSE_EKF_dB_avg_total_original, MSE_EKF_dB_avg_theta_original, MSE_EKF_dB_avg_omega_original, _, _, _,
         _, _] = \
            EKFTest(SysModel=sys_model, test_input=input_test, test_target=target_test, model_AE_conv=None,
                    matrix_data_flag=True, modelKnowledge=modelKnow)

        os.makedirs(
            f"/Simulations_results/Pendulum/EKF/ostia/{modelKnow}_observation_model/",
            exist_ok=True)
        np.savez(
            f"/Simulations_results/Pendulum/EKF/ostia/{modelKnow}_observation_model/v_{v}db_r2_{r2:.1e}_q2_{q2:.1e}.npz",
            MSE_linear_arr_total=MSE_EKF_linear_arr_total.cpu(), MSE_linear_arr_theta=MSE_EKF_linear_arr_theta.cpu(),
            MSE_linear_arr_omega=MSE_EKF_linear_arr_omega.cpu(), MSE_db_avg_total=MSE_EKF_dB_avg_total.cpu(),
            MSE_db_avg_theta=MSE_EKF_dB_avg_theta.cpu(), MSE_db_avg_omega=MSE_EKF_dB_avg_omega.cpu(),
            MSE_EKF_dB_std_total=MSE_EKF_dB_std_total.cpu(), MSE_EKF_dB_std_theta=MSE_EKF_dB_std_theta.cpu(),
            MSE_EKF_dB_std_omega=MSE_EKF_dB_std_omega.cpu(), KG_array=KG_array.cpu(), results_plain=EKF_out.cpu(),
            best_q2=best_q2, MSE_original_total=MSE_EKF_dB_avg_total_original.cpu(),
            MSE_original_theta=MSE_EKF_dB_avg_theta_original.cpu(),
            MSE_original_omega=MSE_EKF_dB_avg_omega_original.cpu())
        print("\n")
