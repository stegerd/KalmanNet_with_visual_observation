import torch
from KalmanNet_sysmdl import SystemModel
from Pipeline_EKF import Pipeline_EKF
from KalmanNet_nn import KalmanNetNN
from model import f, h_full, h_partial, h_nonlinear
import datetime
from datetime import date
import numpy as np
import os

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

ROOT_PATH = "/content/drive/MyDrive/KalmanNet_Visual/KalmanNet_VO/"
ROOT_PATH = r"C:\Users\damis\KalmanNetDrive\KalmanNet_Visual\KalmanNet_VO/"
torch.manual_seed(42)

v = 0
q2 = 1
r2 = 1
m = 2
n = 2
h = h_full
T = 400
T_test = 400
m1x_0 = torch.Tensor([[0.0], [0.0]]).to(dev)
m2x_0 = torch.eye(m).to(dev)
EPOCHS = 20
BATCH_SIZE = 10
LR = 1e-3
WD = 1e-4

today = date.today()
now = datetime.datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)


Q_true = q2 * torch.eye(m)
R_true = r2 * torch.eye(n)
sys_model = SystemModel(f, Q_true, h, R_true, T, T_test)
sys_model.InitSequence(m1x_0, m2x_0)


q = np.sqrt(q2)
r = np.sqrt(r2)
input_ = np.load(ROOT_PATH+f"Datasets/Pendulum/decimated_noisy_data/pendulum_decimated_noisy_q2_{q2:.0e}_r2_{r2:.0e}_v{v}.npz")
target_ = np.load(ROOT_PATH+f"Datasets/Pendulum/decimated_clean_data/pendulum_decimated_q2_{q2:.0e}_v_{v}.npz")
train_input = torch.from_numpy(input_["training_set"][:100, :, :]).float().to(dev)
train_target = torch.from_numpy(target_["training_set"][:100, :, :]).float().to(dev)
test_input = torch.from_numpy(input_["test_set"][:, :, :]).float().to(dev)
test_target = torch.from_numpy(target_["test_set"]).float().to(dev)
cv_input = torch.from_numpy(input_["validation_set"][:100, :, :]).float().to(dev)
cv_target = torch.from_numpy(target_["validation_set"][:100, ...]).float().to(dev)
print(train_input.shape, train_target.shape, cv_input.shape, cv_target.shape)




modelFolder = ROOT_PATH + "Simulations_results/Pendulum/KalmanNet"
os.makedirs(modelFolder, exist_ok=True)
KNet_Pipeline = Pipeline_EKF(strTime, modelFolder, "KNet_vectorial")
KNet_Pipeline.setssModel(sys_model)
KNet_model = KalmanNetNN()
KNet_model.Build(sys_model)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(n_Epochs=EPOCHS, n_Batch=BATCH_SIZE, learningRate=LR, weightDecay=WD)

# KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")
"""
KNet_Pipeline.NNTrain(train_input, train_target, cv_input, cv_target)
[KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(test_input, test_target)
KNet_Pipeline.save()
np.savez(modelFolder+f"result_{q2:.0e}_{r2:.0e}_v{v}.npz", MSE_linear_arr = KNet_MSE_test_linear_arr.cpu(), MSE_linear_avg=KNet_MSE_test_linear_avg.cpu(),
         MSE_dB_avg=KNet_MSE_test_dB_avg.cpu(), output=KNet_test.cpu())
"""
KNet_Pipeline.model.InitSequence(KNet_Pipeline.ssModel.m1x_0, KNet_Pipeline.ssModel.T)

input = train_input[0, :, :]
out = KNet_Pipeline.model(input[:, 6])
print(out)
print(out.shape)