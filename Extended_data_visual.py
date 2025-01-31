import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.nn as nn

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

#######################
### Size of DataSet ###
#######################

# Number of Training Examples
N_E = 5000
# Number of Cross Validation Examples
N_CV = 10
N_T = 100

# Sequence Length for Linear Case
T = 30
T_test = 40

#################
## Design #10 ###
#################
F10 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

H10 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

#  NEEDED ONLY FOR THE SYNTHETIC CASE
H_matrix_for_visual=torch.tensor([[1.0, 0.0],
                                [0.0, 1.0],
                                [0.5, 0.5],
                                [0.75, 0.25],
                                [0.25, 0.75]])
#  NEEDED ONLY FOR THE SYNTHETIC CASE
b_for_visual=torch.tensor([[0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0]])
#  NEEDED ONLY FOR THE SYNTHETIC CASE
class H_fully_connected(nn.Module):
    def __init__(self, weights, bias):
        super(H_fully_connected, self).__init__()
        self.fc = nn.Linear(2, 5)
        with torch.no_grad():
            self.fc.weight.copy_(weights)
            self.fc.bias.copy_(bias.reshape(bias.shape[0]))
        #self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x1 = self.fc(x)
        #x2 = self.fc2(x1)
        return x1

############
## 2 x 2 ###
############
m = 2
n = 2
F = F10[0:m, 0:m]
H = torch.eye(2)
m1_0 = torch.tensor([[0.0], [0.0]]).to(dev)
# m1x_0_design = torch.tensor([[10.0], [-10.0]])
m2_0 = 0 * 0 * torch.eye(m).to(dev)


#############
### 5 x 5 ###
#############
# m = 5
# n = 5
# F = F10[0:m, 0:m]
# H = H10[0:n, 10-m:10]
# m1_0 = torch.zeros(m, 1).to(dev)
# # m1x_0_design = torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [0.0]]).to(dev)
# m2_0 = 0 * 0 * torch.eye(m).to(dev)

##############
## 10 x 10 ###
##############
# m = 10
# n = 10
# F = F10[0:m, 0:m]
# H = H10
# m1_0 = torch.zeros(m, 1).to(dev)
# # m1x_0_design = torch.tensor([[10.0], [-10.0]])
# m2_0 = 0 * 0 * torch.eye(m).to(dev)

# Inaccurate model knowledge based on matrix rotation
alpha_degree = 10
rotate_alpha = torch.tensor([alpha_degree/180*torch.pi]).to(dev)
cos_alpha = torch.cos(rotate_alpha)
sin_alpha = torch.sin(rotate_alpha)
rotate_matrix = torch.tensor([[cos_alpha, -sin_alpha],
                              [sin_alpha, cos_alpha]]).to(dev)
# print(rotate_matrix)
F_rotated = torch.mm(F,rotate_matrix) #inaccurate process model
H_rotated = torch.mm(H,rotate_matrix) #inaccurate observation model

def DataGen_True(SysModel_data, fileName, T):

    SysModel_data.GenerateBatch(1, T, randomInit=False)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    # torch.save({"True Traj":[test_target],
    #             "Obs":[test_input]},fileName)
    torch.save([test_input, test_target], fileName)

def DataGen(SysModel_data, fileName, T, T_test,randomInit=False):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(N_E, T, randomInit=randomInit)
    training_input = SysModel_data.Input
    training_target = SysModel_data.Target

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(N_CV, T, randomInit=randomInit)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(N_T, T_test, randomInit=randomInit)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    #################
    ### Save Data ###
    #################
    torch.save([training_input, training_target, cv_input, cv_target, test_input, test_target], fileName)

def DataLoader(fileName):

    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.load(fileName)
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DataLoader_GPU(fileName):
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.utils.data.DataLoader(torch.load(fileName),pin_memory = False)
    training_input = training_input.squeeze().to(dev).float()
    training_target = training_target.squeeze().to(dev).float()
    cv_input = cv_input.squeeze().to(dev).float()
    cv_target =cv_target.squeeze().to(dev).float()
    test_input = test_input.squeeze().to(dev).float()
    test_target = test_target.squeeze().to(dev).float()
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DecimateData(all_tensors, t_gen,t_mod, offset=0):
    
    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod/t_gen)

    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:,(0+offset)::ratio]
        if(i==0):
            all_tensors_out = torch.cat([tensor], dim=0).view(1,all_tensors.size()[1],-1)
        else:
            all_tensors_out = torch.cat([all_tensors_out,tensor], dim=0)
        i += 1

    return all_tensors_out


def getObs(sequences, h, N_samples, n, T):
    """

    :param sequences: Input sequences
    :param h: Observation matrix/function
    :param N_samples: Number of samples (sequences) in sequences
    :param n: Dimension of observations
    :param T: Length of each sequence
    :return:
    """
    i = 0
    sequences_out = torch.empty(N_samples, n, T)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i, :, t] = h(sequence[:, t])
        i = i+1

    return sequences_out

def Short_Traj_Split(data_target, data_input, T):
    data_target = list(torch.split(data_target, T, 2))
    data_input = list(torch.split(data_input, T, 2))
    data_target.pop()
    data_input.pop()
    data_target = torch.squeeze(torch.cat(list(data_target), dim=0))
    data_input = torch.squeeze(torch.cat(list(data_input), dim=0))
    return [data_target, data_input]
