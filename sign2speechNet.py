import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import dataSetSeq2Seq
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import load_checkpoint, save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_ID_FEAT_SIZE = 0


class sign2speechNet(nn.Module):
    #TODO: Define all of our layers and network shape
    def __init__(self, input_size, hidden_size, num_layers, bidirectional = True):
        super(sign2speechNet, self).__init__()
        self.hidden_size = hidden_size #for lstm
        self.num_layers = num_layers #for lstm
        self.input_size = input_size #Should be set to the number of keypoints

        self.bilstm = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=0.5, #used to be 0.5
                                  bidirectional=bidirectional,
                                  batch_first=True)


    def forward(self, mfcc, hand_keypoints, label):

        #If input is sequential
        #In Shape = Batch * Sequence_Len * Keypoint Shape
        output, (hn, cn) = self.bilstm(hand_keypoints)

        #if input is non sequential
        #In Shape = Batch * Keypoint Shape

        #TODO: Figure out what network architecture we want to use

        #Out Shape = Batch * Sequence_Len * MFCC Features


        return output

######################### Training Set Up #################################

def trainAudio2Landmark(num_epochs = 50000, learning_rate = 0.001, load_model = False, input_size = 63, hidden_size = 128, num_layers = 4, bidirectional = True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Tensorboard
    trainDir = "C:/Users/ionut/Documents/My Python Projects/audio2video/Train_Directory"
    os.chdir(trainDir)
    writer = SummaryWriter('runs/loss_plot')
    step = 0
    valid_step = 0
    model = sign2speechNet(input_size, hidden_size, num_layers, bidirectional).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss() #TODO: Use a better loss funciton than the MSE

    if load_model:
        load_checkpoint(torch.load("my_checkpointTest.pt"), model, optimizer)

    train_mfcc_list = np.load("C:/Users/ionut/Documents/BBC Lip Reading DataSet/TED DataSet Padded Audio and Landmarks/Padded_Audio/padded_audio_array.npy")[:17000]
    train_landmark_list = np.load("C:/Users/ionut/Documents/BBC Lip Reading DataSet/TED DataSet Padded Audio and Landmarks/Padded_Landmarks/padded_landmark_array.npy")[:17000]

    train_dataloader = dataSetSeq2Seq.prepare_dataloader(train_mfcc_list, train_landmark_list, 1)

    validation_landmark_list = np.load("C:/Users/ionut/Documents/BBC Lip Reading DataSet/TED DataSet Padded Audio and Landmarks/Padded_Landmarks/padded_landmark_array.npy")[17000:]
    validation_mfcc_list = np.load("C:/Users/ionut/Documents/BBC Lip Reading DataSet/TED DataSet Padded Audio and Landmarks/Padded_Audio/padded_audio_array.npy")[17000:]
    validation_dataloader = dataSetSeq2Seq.prepare_dataloader(validation_mfcc_list, validation_landmark_list, 1)

    min_valid_loss = np.inf

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

#########################Train Loop################################################

        train_loss = 0
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            mfcc, hand_keypoints, label = batch
            mfcc = mfcc.float().to(device)
            hand_keypoints = hand_keypoints.float().to(device)
            label = label.float().to(device)

            predicted_mfcc = model(mfcc, hand_keypoints, label)
            loss = criterion(predicted_mfcc, mfcc)

            optimizer.zero_grad()
            loss.backward()

            # Grad Descent Step
            optimizer.step()

            # Plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            train_loss += loss.item()
            step += 1
            print('step = '+str(step))

#########################Validation Loop############################################

        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer
        for batch_idx, batch in enumerate(validation_dataloader):
            mfcc, hand_keypoints, label = batch
            mfcc = mfcc.float().to(device)
            hand_keypoints = hand_keypoints.float().to(device)
            label = label.float().to(device)

            # Forward Pass
            predicted_mfcc = model(mfcc, hand_keypoints, label)

            # Find the Loss
            loss = criterion(predicted_mfcc, mfcc)

            writer.add_scalar("Validation Loss", loss, global_step=valid_step)
            valid_step += 1
            # Calculate Loss
            valid_loss += loss.item()

        print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(validation_dataloader)}')
        writer.add_scalar("Overall Valid Loss", (valid_loss / len(validation_dataloader)), global_step=epoch)
        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            # Saving State Dict
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint)


######################### Inference Phase #################################









