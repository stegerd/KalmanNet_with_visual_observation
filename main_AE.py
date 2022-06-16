import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from Extended_data_visual import N_E, N_CV, N_T, T, T_test
from visual_supplementary import create_dataset, visualize_similarity, y_size
from Extended_data_visual import DataLoader_GPU
import numpy as np
import json

def train_conv(encoder, decoder, train_loader, val_loader, loss_fn, optimizer, num_epochs,flag_only_encoder):
    # Set train mode for both the encoder and the decoder
    train_loss_epoch=[]
    val_loss_epoch = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for epoch in range(num_epochs):
        train_loss = []
        encoder.train()
        decoder.train()
        for k,batch in enumerate(train_loader):  # with "_" we just ignore the labels (the second element of the dataloader tuple)
            if flag_only_encoder:
                image_batch = batch[0]
                targets_batch = batch[1].reshape(-1,1)
            else:
                image_batch = batch[0]
                targets_batch = batch[0]
            if(image_batch.shape[0]==256):
                encoded_data = encoder(image_batch.reshape(256,1,24,24)/255)
                if flag_only_encoder:
                    decoded_data = encoded_data
                    loss = loss_fn(decoded_data.float(), targets_batch.float())
                else:
                    decoded_data = decoder(encoded_data)
                    if(k==10):
                        check_learning_process(targets_batch.reshape(256,1,24,24)/255, decoded_data, epoch, 'train')
                    loss = loss_fn(decoded_data, targets_batch.reshape(256, 1, 24, 24) / 255)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print batch loss
                train_loss.append(loss.detach().cpu().numpy())
        train_loss_epoch.append(np.mean(train_loss).item())
        epoch_val_loss=test_epoch(encoder, decoder, val_loader, loss_fn, epoch, flag_only_encoder)
        val_loss_epoch.append(epoch_val_loss)
        print('epoch: {} train loss: {} val loss: {}'.format(epoch,np.mean(train_loss),epoch_val_loss))
    return encoder, decoder, val_loss_epoch, train_loss_epoch


def test_encoder(encoder, test_input, test_target):

    MSE_test_linear_arr = torch.empty([test_input.shape[0]])

    encoder.eval()
    loss_fn = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        """
        for sequence in range(test_input.shape[0]):
            x_out_test = torch.empty(1, test_input.shape[1])
            y = test_input[sequence,...]
            for t in range(test_input.shape[1]):
                AE_input = y[t, :, :].reshape(1, 1, 24, 24) / 255
                x_out_test[:, t] = encoder(AE_input)
            MSE_test_linear_arr[sequence] = loss_fn(x_out_test, test_target[sequence, 0, :]).item()
        """
    test_input = test_input.reshape((test_input.shape[0]*test_input.shape[1], test_input.shape[2], test_input.shape[2]))
    test_target = test_target.reshape((test_target.shape[0]*test_target.shape[1], test_target.shape[2]))
    input_ae = torch.unsqueeze(test_input, 1)
    output = encoder(input_ae/255)
    print(f"output shape: {output.shape}")
    MSE= loss_fn(output, test_target[:, 0:1])
    MSE_log =  10 * torch.log10(MSE)
    new_line='\n'
    print(f"MSE TEST ENCODER-ONLY LINEAR-SCALE:{MSE.item()} {new_line}"
          f"MSE TEST ENCODER-ONLY LOG-SCALE: {MSE_log} dB ")
    #return MSE_test_linear_arr, MSE_linear_avg, MSE_log_avg

def test_epoch(encoder, decoder, val_loader, loss_fn, epoch,  flag_only_encoder):
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        val_loss = []
        for k,batch in enumerate(val_loader):
            image_batch=batch[0]
            targets_batch=batch[1].reshape(-1,1)
            if True:#(image_batch.shape[0] == 256): #taking only full batch
                encoded_data = encoder(image_batch.reshape(-1, 1, 24, 24) / 255)
                if flag_only_encoder:
                    decoded_data = encoded_data
                    loss = loss_fn(decoded_data.float(), targets_batch.float())
                else:
                    decoded_data = decoder(encoded_data)
                    if (k == 10):
                        check_learning_process(image_batch.reshape(256, 1, 24, 24) / 255, decoded_data, epoch, 'val')
                    loss = loss_fn(decoded_data, targets_batch.reshape(256, 1, 24, 24) / 255)
                val_loss.append(loss.detach().cpu().numpy())
    return np.mean(val_loss).item()

def train(model, H_data_train,H_data_valid, num_epochs, batch_size, learning_rate):
    torch.manual_seed(42)
    criterion_y = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_loader = torch.utils.data.DataLoader(H_data_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(H_data_valid, batch_size=len(H_data_valid), shuffle=True)
    outputs = []
    epoch_list = []
    Loss_train_list = []
    Loss_valid_list = []
    for epoch in range(num_epochs):
        for data in train_loader:
            state_batch, img_batch = data
            x_bottom_batch,recon_batch = model(img_batch)
            #check_learning_process(img_batch, recon_batch, epoch, 'Train')
            loss_y = criterion_y(recon_batch, img_batch)
            #loss_x = criterion_x(x_bottom_batch, state_batch)
            #Total_loss = loss_y + loss_x
            loss_y.backward()
            optimizer.step()
            optimizer.zero_grad()

            #validation
            for data in val_loader:
                state_batch, img_batch = data
                x_bottom_batch, recon_batch = model(img_batch)
                #check_learning_process(img_batch, recon_batch, epoch, 'Val')
                loss_y_valid = criterion_y(recon_batch, img_batch)

        epoch_list.append(epoch)
        Loss_valid_list.append(float(loss_y_valid))
        Loss_train_list.append(float(loss_y))
        print('Epoch:{}, Train Loss:{:.7f}, Valid Loss:{:.7f}'.format(epoch+1, float(loss_y), float(loss_y_valid)))
    outputs.append((state_batch, x_bottom_batch, img_batch, recon_batch))
    return outputs, epoch_list, Loss_train_list, Loss_valid_list, model

#Fully connected autoencoder for visual synthetic data
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(y_size * y_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5)
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, y_size * y_size),
            torch.nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x_bottom = x
        y_rec = self.decoder(x_bottom)
        return x_bottom,y_rec

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder, self).__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True))

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, encoded_space_dim))

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Decoder, self).__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, 128),
            nn.ReLU(True))
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(32, 2, 2))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(8, 1, 3, stride=2,padding=1, output_padding=1))

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        #x = torch.sigmoid(x)
        return x

def check_learning_process(img_batch,recon_batch,epoch, name):
    y_nump = img_batch[32].reshape(24,24).detach().numpy().squeeze()
    y_recon_nump = recon_batch[32].reshape(24,24).detach().numpy().squeeze()
    #y_nump = img_batch[10][5]
    #y_recon_nump = recon_batch[10][5]
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(y_nump, cmap='gray')
    plt.axis('off')
    plt.title("origin")

    fig.add_subplot(1, 2, 2)
    plt.imshow(y_recon_nump, cmap='gray')
    plt.axis('off')
    plt.title("reconstruct")
    fig.savefig('AE Process/{} Process at epoch {}.PNG'.format(name, epoch))

#USED ONLY FOR SYNTETHIC VISUAL DATA
def train_AE_FC():
    dataFolderName = 'Simulations/Synthetic_visual' + '/'
    dataFileName = 'y{}x{}_Ttrain{}_NE{}_NCV{}_NT{}_Ttest{}_Sigmoid.pt'.format(y_size,y_size, T,N_E,N_CV,N_T,T_test)
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName)
    H_data_train = create_dataset(train_input, train_target, N_E)
    H_data_valid = create_dataset(cv_input, cv_target, N_CV)
    model = Autoencoder()
    max_epochs = 20
    BATCH_ZISE = 128
    LR = 1e-3
    outputs, epoch_list, Loss_train_list, Loss_valid_list, AE_model = train(model, H_data_train,H_data_valid, num_epochs=max_epochs, batch_size=BATCH_ZISE, learning_rate=LR)
    return AE_model

    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 1, 1)
    plt.plot(epoch_list, Loss_train_list, label='Train')
    plt.plot(epoch_list, Loss_valid_list, label='Val')
    plt.title("AutoEncoader Loss")
    fig.savefig('AE Process/Loss')

    visualize_similarity(outputs[0][2][22].squeeze().detach().numpy().reshape((y_size,y_size)), outputs[0][3][22].squeeze().detach().numpy().reshape((y_size,y_size)))

def print_process(val_loss_list, train_loss_list , flag_only_encoder, path_to_save):
    if flag_only_encoder:
        title='encoder only'
    else:
        title='Auto Encoder'
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 1, 1)
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
    plt.title("Loss of " + title)
    plt.legend()
    plt.savefig(path_to_save + f"Learning_process {title}")

def load_pendulum_npz(path):
    file = np.load(path)
    file_np = file.f.images
    return file_np

def training_AE_conv1(flag_only_encoder):
    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(r'./Simulations/Pendulum/y24x24_Ttrain30_NE1000_NCV100_NT100_Ttest40_pendulum_train_encoder.pt')


    imgs_np=train_input.cpu().detach().numpy()
    states_np=train_target.cpu().detach().numpy()
    train_list=[]
    for k in range(imgs_np.shape[0]):
        sample=imgs_np[k]
        for t in range(sample.shape[0]):
            img=sample[t]
            train_list.append(img)

    if flag_only_encoder:
        chosen_targets_np= states_np
        targets_list = []
        for k in range(chosen_targets_np.shape[0]):
            sample = chosen_targets_np[k]
            for t in range(sample.shape[0]):
                target=sample[t,0:1] #learn only the angle, since angular velocity is hard to learn from single image
                targets_list.append(target)
    else:
        targets_list=train_list

    lr = 1e-3
    wd = 1e-4
    torch.manual_seed(0)
    d = 1 #learn only the angle
    num_epochs=500
    batch_size=256
    N_train= int(len(train_list)*0.7)
    N_val = int((len(train_list)-N_train)/2)

    data_train = train_list[:N_train]
    targets_train = targets_list[:N_train]
    dataset_train=[]
    data_val_test = train_list[N_train:]
    targets_val_test = targets_list[N_train:]

    data_val = data_val_test[:N_val]
    targets_val = targets_val_test[:N_val]
    dataset_val = []
    data_test = data_val_test[N_val:]
    targets_test = targets_val_test[N_val:]
    dataset_test = []

    for k in range(len(data_train)):
        dataset_train.append((data_train[k],targets_train[k]))
    for k in range(len(data_val)):
        dataset_val.append((data_val[k],targets_val[k]))
    for k in range(len(data_test)):
        dataset_test.append((data_test[k], targets_test[k]))

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

    loss_fn = torch.nn.MSELoss()
    encoder = Encoder(encoded_space_dim=d)
    decoder = Decoder(encoded_space_dim=d)
    params_to_optimize = [{'params': encoder.parameters()},{'params': decoder.parameters()}]
    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
    encoder, decoder, val_loss_epoch, train_loss_epoch = train_conv(encoder, decoder, train_loader,valid_loader, loss_fn, optim, num_epochs,flag_only_encoder)
    print_process(val_loss_epoch, train_loss_epoch, flag_only_encoder)
    return encoder, decoder, test_loader

def training_AE_conv(flag_only_encoder, path_to_save):
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        dev = torch.device("cpu")

    [train_input, train_target, cv_input, cv_target, test_input, test_target] = \
        DataLoader_GPU(r'Datasets/Pendulum/pendulum_trans_std_0.001_obs_std_1e-05_train_5000&30_val_100&30_test_100&40.pt')
    train_input = torch.reshape(train_input, (train_input.shape[0]*train_input.shape[1], 24, 24))
    #train_target = torch.transpose(train_target, 1, 2)
    train_target = torch.reshape(train_target, (train_target.shape[0] * train_target.shape[1], 2))
    cv_input = torch.reshape(cv_input, (cv_input.shape[0] * cv_input.shape[1], 24, 24))
    #cv_target = torch.transpose(cv_target, 1, 2)
    cv_target = torch.reshape(cv_target, (cv_target.shape[0] * cv_target.shape[1], 2))
    test_input = torch.reshape(test_input, (test_input.shape[0] * test_input.shape[1], 24, 24))
   # test_target = torch.transpose(test_target, 1, 2)
    test_target = torch.reshape(test_target, (test_target.shape[0] * test_target.shape[1], 2))

    if flag_only_encoder:
        train_target = train_target[:, 0]
        cv_target = cv_target[:, 0]
        test_target = test_target[:, 0]

    dataset_train = torch.utils.data.TensorDataset(train_input, train_target)
    dataset_val = torch.utils.data.TensorDataset(cv_input, cv_target)
    dataset_test = torch.utils.data.TensorDataset(test_input, test_target)

    lr = 1e-3
    wd = 1e-4
    torch.manual_seed(0)
    d = 1 #learn only the angle
    num_epochs=500
    batch_size=256

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

    loss_fn = torch.nn.MSELoss()
    encoder = Encoder(encoded_space_dim=d).to(dev, non_blocking=True)
    decoder = Decoder(encoded_space_dim=d).to(dev, non_blocking=True)
    params_to_optimize = [{'params': encoder.parameters()},{'params': decoder.parameters()}]
    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
    encoder, decoder, val_loss_epoch, train_loss_epoch = train_conv(encoder, decoder, train_loader, valid_loader,
                                                                    loss_fn, optim, num_epochs, flag_only_encoder)
    print_process(val_loss_epoch, train_loss_epoch, flag_only_encoder, path_to_save)
    losses_and_hyperparameters = {'epochs': num_epochs, 'batch_size': batch_size, 'learning_rate': lr,
                                  'weight_decay': wd, 'encoder_dim': d, 'training_loss': train_loss_epoch,
                                  'validation_loss': val_loss_epoch}
    with open(path_to_save+'losses_and_hyperparameters.txt', 'w') as file:
        file.write(json.dumps(losses_and_hyperparameters))
    return encoder, decoder, test_loader

if __name__ == '__main__':
    flag_only_encoder = 1
    path_to_save = "./AE process/training_new_big_dataset/"
    os.makedirs(path_to_save, True)

    encoder, decoder, test_loader = training_AE_conv(flag_only_encoder=1, path_to_save=path_to_save)
    os.makedirs("./saved_models/training_new_big_dataset/", True)
    if flag_only_encoder:
        torch.save(encoder.state_dict(), r"./saved_models/training_new_big_dataset/only_conv_encoder.pt")









    """
    flag_only_encoder=True
    encoder, decoder, test_loader = training_AE_conv(flag_only_encoder)
    if flag_only_encoder:
        torch.save(encoder.state_dict(), r"./saved_models/Only_conv_encoder.pt")
    else:
        torch.save(encoder.state_dict(), r"./saved_models/AutoEncoder_conv_encoder.pt")
        torch.save(decoder.state_dict(), r"./saved_models/AutoEncoder_conv_decoder.pt")

    encoder_loaded = Encoder(5)
    encoder_loaded.load_state_dict(torch.load(r"./saved_models/AutoEncoder_conv_encoder.pt"))
    decoder_loaded = Decoder(5)
    decoder_loaded.load_state_dict(torch.load(r"./saved_models/AutoEncoder_conv_decoder.pt"))

    test_loss = test_epoch(encoder_loaded, decoder_loaded, test_loader, torch.nn.MSELoss(), 1, flag_only_encoder)
    print("Test loss is: {}".format(test_loss))
    """





