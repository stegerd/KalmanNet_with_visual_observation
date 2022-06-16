import torch
import torch.nn as nn
import random
from Plot import Plot
import time

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("using GPU!")
else:
    dev = torch.device("cpu")
    print("using CPU!")


class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model.to(dev, non_blocking=True)

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch  # Number of Samples in Batch
        self.learningRate = learningRate  # Learning Rate
        self.weightDecay = weightDecay  # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def NNTrain(self, train_input, train_target, cv_input, cv_target, save_model_wrt, weight_theta_loss=0.5, weight_omega_loss=0.5):
        cv_target = cv_target.to(dev, non_blocking=True)
        train_target = train_target.to(dev, non_blocking=True)

        self.N_E = train_input.size()[0]
        self.N_CV = cv_input.size()[0]

        MSE_cv_linear_batch = torch.empty([3, self.N_CV]).to(dev, non_blocking=True)
        self.MSE_cv_linear_epoch = torch.empty([3, self.N_Epochs]).to(dev, non_blocking=True)
        self.MSE_cv_dB_epoch = torch.empty([3, self.N_Epochs]).to(dev, non_blocking=True)

        MSE_train_linear_batch = torch.empty([3, self.N_B]).to(dev, non_blocking=True)
        self.MSE_train_linear_epoch = torch.empty([3, self.N_Epochs]).to(dev, non_blocking=True)
        self.MSE_train_dB_epoch = torch.empty([3, self.N_Epochs]).to(dev, non_blocking=True)



        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_Epochs):

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            for j in range(0, self.N_CV):
                y_cv = cv_input[j, :, 1:]
                self.model.InitSequence(cv_target[j, :, 0], self.ssModel.T_cv-1)

                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T_cv-1)
                for t in range(0, self.ssModel.T_cv-1):
                    x_out_cv[:, t] = self.model(y_cv[:, t])

                # Compute Validation Loss
                MSE_cv_linear_batch[0, j] = weight_theta_loss * self.loss_fn(x_out_cv[0], cv_target[j, 0, 1:]).item() \
                                            + weight_omega_loss * self.loss_fn(x_out_cv[1], cv_target[j, 1, 1:]).item()
                MSE_cv_linear_batch[1, j] = self.loss_fn(x_out_cv[0], cv_target[j, 0, 1:]).item()
                MSE_cv_linear_batch[2, j] = self.loss_fn(x_out_cv[1], cv_target[j, 1, 1:]).item()

            # Average
            self.MSE_cv_linear_epoch[0, ti] = torch.mean(MSE_cv_linear_batch[0])
            self.MSE_cv_linear_epoch[1, ti] = torch.mean(MSE_cv_linear_batch[1])
            self.MSE_cv_linear_epoch[2, ti] = torch.mean(MSE_cv_linear_batch[2])
            self.MSE_cv_dB_epoch[0, ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[0, ti])
            self.MSE_cv_dB_epoch[1, ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[1, ti])
            self.MSE_cv_dB_epoch[2, ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[2, ti])

            if (self.MSE_cv_dB_epoch[save_model_wrt, ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[save_model_wrt, ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                n_e = random.randint(0, self.N_E - 1)

                y_training = train_input[n_e, :, 1:]
                self.model.InitSequence(train_target[n_e, :, 0], self.ssModel.T-1)

                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T-1).to(dev, non_blocking=True)
                for t in range(0, self.ssModel.T-1):
                    x_out_training[:, t] = self.model(y_training[:, t])

                # Compute Training Loss
                LOSS = weight_theta_loss * self.loss_fn(x_out_training[0], train_target[n_e, 0, 1:]) \
                       + weight_omega_loss * self.loss_fn(x_out_training[1], train_target[n_e, 1, 1:])

                MSE_train_linear_batch[0, j] = LOSS.item()
                MSE_train_linear_batch[1, j] = self.loss_fn(x_out_training[0], train_target[n_e, 0, 1:]).item()
                MSE_train_linear_batch[2, j] = self.loss_fn(x_out_training[1], train_target[n_e, 1, 1:]).item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[0, ti] = torch.mean(MSE_train_linear_batch[0])
            self.MSE_train_linear_epoch[1, ti] = torch.mean(MSE_train_linear_batch[1])
            self.MSE_train_linear_epoch[2, ti] = torch.mean(MSE_train_linear_batch[2])
            self.MSE_train_dB_epoch[0, ti] = 10 * torch.log10(self.MSE_train_linear_epoch[0, ti])
            self.MSE_train_dB_epoch[1, ti] = 10 * torch.log10(self.MSE_train_linear_epoch[1, ti])
            self.MSE_train_dB_epoch[2, ti] = 10 * torch.log10(self.MSE_train_linear_epoch[2, ti])

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
            print(f"{ti}, MSE Training total/theta/omega: {self.MSE_train_dB_epoch[0, ti]:.2f}/{self.MSE_train_dB_epoch[1, ti]:.2f}"
                  f"/{self.MSE_train_dB_epoch[2, ti]:.2f} [dB], MSE Validation total/theta/omega :{self.MSE_cv_dB_epoch[0, ti]:.2f}"
                  f"/{self.MSE_cv_dB_epoch[1, ti]:.2f}/{self.MSE_cv_dB_epoch[2, ti]:.2f} [dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[:, ti] - self.MSE_train_dB_epoch[:, ti - 1]
                d_cv = self.MSE_cv_dB_epoch[:, ti] - self.MSE_cv_dB_epoch[:, ti - 1]
                print(f"diff MSE Training total/theta/cv : {d_train[0]:.2f}/{d_train[1]:.2f}/{d_train[2]:.2f} [dB], "
                      f"diff MSE Validation total/theta/cv: {d_cv[0]:2f}/{d_cv[1]:2f}/{d_cv[2]:2f} [dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

    def NNTest(self, test_input, test_target):
        test_target = test_target.to(dev, non_blocking=True)
        self.N_T = test_input.size()[0]

        self.MSE_test_linear_arr_total = torch.empty([self.N_T], requires_grad=False)
        self.MSE_test_linear_arr_theta = torch.empty([self.N_T], requires_grad=False)
        self.MSE_test_linear_arr_omega = torch.empty([self.N_T], requires_grad=False)

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        self.model = torch.load(self.modelFileName, map_location=dev)

        self.model.eval()



        x_out_array = torch.empty(self.N_T, self.ssModel.m, self.ssModel.T_test - 1)
        start = time.time()
        for j in range(0, self.N_T):
            with torch.no_grad():

                y_mdl_tst = test_input[j, :, 1:]

                self.model.InitSequence(test_target[j, :, 0], self.ssModel.T_test - 1)

                x_out_test = torch.empty(self.ssModel.m, self.ssModel.T_test - 1).to(dev, non_blocking=True)

                for t in range(0, self.ssModel.T_test - 1):
                    x_out_test[:, t] = self.model(y_mdl_tst[:, t])



                self.MSE_test_linear_arr_total[j] = loss_fn(x_out_test, test_target[j, :, 1:]).item()
                self.MSE_test_linear_arr_theta[j] = loss_fn(x_out_test[0], test_target[j, 0, 1:]).item()
                self.MSE_test_linear_arr_omega[j] = loss_fn(x_out_test[1], test_target[j, 1, 1:]).item()

                x_out_array[j, :, :] = x_out_test

        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg_total = torch.mean(self.MSE_test_linear_arr_total)
        self.MSE_test_dB_avg_total = 10 * torch.log10(self.MSE_test_linear_avg_total)

        self.MSE_test_linear_avg_theta = torch.mean(self.MSE_test_linear_arr_theta)
        self.MSE_test_dB_avg_theta = 10 * torch.log10(self.MSE_test_linear_avg_theta)

        self.MSE_test_linear_avg_omega = torch.mean(self.MSE_test_linear_arr_omega)
        self.MSE_test_dB_avg_omega = 10 * torch.log10(self.MSE_test_linear_avg_omega)

        # Standard deviation
        self.MSE_test_std_total = torch.std(self.MSE_test_linear_arr_total, unbiased=True)
        self.MSE_test_dB_std_total = 10 * torch.log10(
            self.MSE_test_linear_avg_total + self.MSE_test_std_total) - self.MSE_test_dB_avg_total

        self.MSE_test_std_theta = torch.std(self.MSE_test_linear_arr_theta, unbiased=True)
        self.MSE_test_dB_std_theta = 10 * torch.log10(
            self.MSE_test_linear_avg_theta + self.MSE_test_std_theta) - self.MSE_test_dB_avg_theta

        self.MSE_test_std_omega = torch.std(self.MSE_test_linear_arr_omega, unbiased=True)
        self.MSE_test_dB_std_omega = 10 * torch.log10(
            self.MSE_test_linear_avg_omega + self.MSE_test_std_omega) - self.MSE_test_dB_avg_omega

        # Print MSE Cross Validation

        print(f" Test mse total/theta/omega: {self.MSE_test_dB_avg_total}/{self.MSE_test_dB_avg_theta}/{self.MSE_test_dB_avg_omega}")
        # Print std
        print(f" Test STD total/theta/omega: {self.MSE_test_dB_std_total}/{self.MSE_test_dB_std_theta}/{self.MSE_test_dB_std_omega}")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr_total, self.MSE_test_dB_avg_total, self.MSE_test_dB_std_total,
                self.MSE_test_linear_arr_theta, self.MSE_test_dB_avg_theta, self.MSE_test_dB_std_theta,
                self.MSE_test_linear_arr_omega, self.MSE_test_dB_avg_omega, self.MSE_test_dB_std_omega, x_out_array]

    def PlotTrain_KF(self, MSE_KF_linear_arr=None, MSE_KF_dB_avg=None):

        self.Plot = Plot(self.folderName, self.modelName)


        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg_total, self.MSE_cv_dB_epoch[0], self.MSE_train_dB_epoch[0], loss_type="total")
        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg_theta, self.MSE_cv_dB_epoch[1], self.MSE_train_dB_epoch[1], loss_type="theta")
        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg_omega, self.MSE_cv_dB_epoch[2], self.MSE_train_dB_epoch[2], loss_type="omega")



        #self.Plot.NNPlot_Hist(self.MSE_test_linear_arr_total, MSE_KF_linear_arr)