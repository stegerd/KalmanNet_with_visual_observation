import torch
import torch.nn as nn
import random
from Plot import Plot
import time
import matplotlib.pyplot as plt
import copy

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

class Pipeline_KF:

    def __init__(self, Time, folderName, modelName, data_name):
        super().__init__()
        self.Time = Time
        self.folderName = folderName #+ '/'
        self.modelName = modelName + '_' + data_name
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"


    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, KNet_model):
        self.model = KNet_model

    def setTrainingParams(self, fix_H_flag, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.fix_H_flag= fix_H_flag


    def NNTrain(self, train_input, train_target, cv_input, cv_target, model_AE_conv, matrix_data_flag):

        self.N_E = train_input.size()[0]
        self.N_CV = cv_input.size()[0]

        MSE_cv_linear_batch = torch.empty([self.N_CV]).to(dev, non_blocking=True)
        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)

        MSE_train_linear_batch = torch.empty([self.N_B]).to(dev, non_blocking=True)
        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        Train_loss_list = []
        Val_loss_list = []
        for ti in range(0, self.N_Epochs):
            t = time.time()
            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            for j in range(0, self.N_CV):
                self.model.InitSequence(self.ssModel.m1x_0)
                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T)

                if matrix_data_flag:
                    y_cv = cv_input[j, :, :]
                    for t in range(0, self.ssModel.T):
                        x_out_cv[:, t] = self.model(y_cv[:, t], self.fix_H_flag)
                else:
                    y_cv = cv_input[j, :, :, :]
                    for t in range(0, self.ssModel.T):
                        AE_input = y_cv[t, :, :].reshape(1,1,24,24)/255
                        y_cv_decoaded_t = model_AE_conv(AE_input)
                        x_out_cv[:, t] = self.model(y_cv_decoaded_t, self.fix_H_flag)

                # Compute Training Loss
                MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j, :, :]).item()

            # Average
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)
                #self.best_model = copy.deepcopy(self.model)

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # Training Mode
            self.model.train()
            # Init Hidden State
            self.model.init_hidden()
            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                self.model.InitSequence(self.ssModel.m1x_0)
                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T)
                n_e = random.randint(0, self.N_E - 1)

                if matrix_data_flag:
                    y_training = train_input[n_e, :, :]
                    for t in range(0, self.ssModel.T):
                        x_out_training[:, t] = self.model(y_training[:, t], self.fix_H_flag)
                else:
                    y_training = train_input[n_e, :, :, :]
                    for t in range(0, self.ssModel.T):
                        AE_input = y_training[t, :, :].reshape(1,1,24,24)/255
                        y_training_decoaded_t = model_AE_conv(AE_input)
                        x_out_training[:, t] = self.model(y_training_decoaded_t, self.fix_H_flag)

                # Compute Training Loss
                LOSS = self.loss_fn(x_out_training.float(), train_target[n_e, :, :].float())
                MSE_train_linear_batch[j] = LOSS.item()
                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti], "[dB] , timing ", time.time() - t)
            # if (self.MSE_cv_dB_epoch[ti]>12 and ti>100):
            #     print("configuration is not good enough")
            #     break
            Train_loss_list.append(self.MSE_train_dB_epoch[ti].cpu().numpy())
            Val_loss_list.append(self.MSE_cv_dB_epoch[ti].cpu().numpy())
            if ti > 1:
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                #print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")
        return Train_loss_list, Val_loss_list
        #self.PlotTrain_KF(self.MSE_train_linear_epoch, self.MSE_train_dB_epoch)
        #self.print_process(Val_loss_list, Train_loss_list, title)

    def print_process(self, val_loss_list, train_loss_list, title):
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1, 1, 1)
        plt.plot(train_loss_list, 'r', label='train')
        plt.plot(val_loss_list, 'g', label='val')
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title("Loss of {}".format(title))
        plt.legend()
        plt.savefig(self.folderName + self.modelName + "_loss_plot.png")

    def NNTest(self, n_Test, test_input, test_target, model_AE, model_AE_conv, matrix_data_flag):

        self.N_T = n_Test
        self.MSE_test_linear_arr = torch.empty([self.N_T])
        self.MSE_test_theta_linear_arr = torch.empty([self.N_T])
        self.MSE_test_omega_linear_arr = torch.empty([self.N_T])
        prediction = torch.empty([n_Test, self.ssModel.m, self.ssModel.T_test])

    # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')
        self.model = torch.load(self.modelFileName, map_location=dev)
        self.model = self.model.to(dev, non_blocking=True)
        self.model.eval()
        torch.no_grad()
        start = time.time()

        for j in range(0, self.N_T):
            self.model.InitSequence(self.ssModel.m1x_0)
            x_out_test = torch.empty(self.ssModel.m, self.ssModel.T_test)

            if matrix_data_flag:
                y_mdl_tst = test_input[j, :, :]
                for t in range(0, self.ssModel.T_test):
                    x_out_test[:, t] = self.model(y_mdl_tst[:, t], self.fix_H_flag)
            else:
                y_mdl_tst = test_input[j, :, :, :]
                for t in range(0, self.ssModel.T_test):
                    AE_input = y_mdl_tst[t, :, :].reshape(1, 1, 24, 24) / 255
                    y_test_decoaded_t = model_AE_conv(AE_input)
                    x_out_test[:, t] = self.model(y_test_decoaded_t.to(dev, non_blocking=True), self.fix_H_flag)

            prediction[j, ...] = x_out_test

            # Compute Training Loss
            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j, :, :]).item()
            self.MSE_test_theta_linear_arr[j] = loss_fn(x_out_test[0, :], test_target[j, 0, :]).item()
            self.MSE_test_omega_linear_arr[j] = loss_fn(x_out_test[1, :], test_target[j, 1, :]).item()
        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_theta_linear_avg = torch.mean(self.MSE_test_theta_linear_arr)
        self.MSE_test_omega_linear_avg = torch.mean(self.MSE_test_omega_linear_arr)

        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        self.MSE_test_theta_dB_avg = 10 * torch.log10(self.MSE_test_theta_linear_avg)
        self.MSE_test_omega_dB_avg = 10 * torch.log10(self.MSE_test_omega_linear_avg)




    # Standard deviation
        self.MSE_test_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_linear_avg + self.MSE_test_std) - self.MSE_test_dB_avg

        new_line = '\n'
        # Print MSE on test set
        str = self.modelName + "-" + "MSE Test log-scale:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test log-scale:"
        print(str, self.MSE_test_dB_std, "[dB]")
        print(f"MSE total linear|log: {self.MSE_test_linear_avg}|{self.MSE_test_dB_avg}dB {new_line} MSE theta linear|log: {self.MSE_test_theta_linear_avg}|{self.MSE_test_theta_dB_avg}dB {new_line}"
              f"MSE omega linear|log: {self.MSE_test_omega_linear_avg}|{self.MSE_test_omega_dB_avg}dB,")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_avg, self.MSE_test_theta_linear_avg,
                self.MSE_test_omega_linear_avg, self.MSE_test_dB_avg, prediction]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg, self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)