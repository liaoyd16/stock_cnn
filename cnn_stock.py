
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

import os
import json

# training & testing meta-param
EPOCH = 300
BS = 5
DATE_LEN = 52
SEQ_LEN = 40
INPUT_LEN = 32
OUTPUT_LEN = 8

LAMB = 0.1

THETA = .004

# freature vector
FEATURES = 7
## name2index
TIMESTAMP = 0
P0 = 1
P_low = 2
P_high = 3
Pt = 4
P_average = 5
VOL = 6
TURNOVER = 7

class CNN_stock(nn.Module):

    def __init__(self):
        super(CNN_stock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(FEATURES, 8, kernel_size=7, padding=3),
            nn.Conv1d(8, 8, kernel_size=1),
        )
            
        self.bn1 = nn.BatchNorm1d(8)
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16,kernel_size=7, padding=3),
            nn.Conv1d(16,16,kernel_size=1),
        )
        self.mp2 = nn.Conv1d(16,16,kernel_size=2,stride=2)
        self.bn2 = nn.BatchNorm1d(16)
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(16,32,kernel_size=7, padding=3),
            nn.Conv1d(32,32,kernel_size=1),
        )
        self.mp3 = nn.Conv1d(32,32,kernel_size=2,stride=2)
        self.bn3 = nn.BatchNorm1d(32)

        self.lin_encode = nn.Linear(256, 128)
        self.lin_decode = nn.Linear(128, 128)

        self.demp3 = nn.ConvTranspose1d(32,32,kernel_size=2,stride=2)
        self.deconv3 = nn.Sequential(
            nn.Conv1d(32,32,kernel_size=1),
            nn.ConvTranspose1d(32,16,kernel_size=7, padding=3),
        )
        self.debn3 = nn.BatchNorm1d(16)

        self.deconv2 = nn.Sequential(
            nn.Conv1d(16,16,kernel_size=1),
            nn.ConvTranspose1d(16, 8,kernel_size=7, padding=3)
        )
        self.debn2 = nn.BatchNorm1d(8)

        self.deconv1 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=1),
            nn.ConvTranspose1d(8, 2, kernel_size=7, padding=3)
        )
        self.debn1 = nn.BatchNorm1d(2)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        
        x = self.mp2(F.relu(self.conv2(x)))
        x = self.bn2(x)

        x = self.mp3(F.relu(self.conv3(x)))
        x = self.bn3(x)

        x = x.reshape(BS, 256)
        x = F.relu(self.lin_encode(x))
        x = F.relu(self.lin_decode(x))
        x = x.reshape(BS, 32, 4)

        x = self.demp3(x)
        x = F.relu(self.deconv3(x))
        x = self.debn3(x)

        x = F.relu(self.deconv2(x))
        x = self.debn2(x)

        x = F.relu(self.deconv1(x))
        x = self.debn1(x)

        return x



# def lossF(outputs, answer):
#     print(outputs.shape, answer.shape)
#     return nn.MSELoss(outputs - answer[:,[P0,Pt]])


def __get_trends(vector_batch):
    trends = [(v[Pt] - v[P0] / v[P0]) for v in batch]
    trends[abs(trends) < THETA] = 0
    trends[trends > 0] = 1
    trends[trends < 0] = 2
    return trends
    # returns a [0/1/2] batch, 0 for ->, 1 for ↑, 2 for ↓


if __name__ == '__main__':
    cnn = CNN_stock()
    optimizer = torch.optim.SGD(cnn.parameters(), momentum=0.9, lr=.5)
    lossF = nn.MSELoss()

    # dataset
    TEST = "./TickJson/test/"
    TRAIN = "./TickJson/train/"
    testset = []
    for file in os.listdir(TEST):
        if '.json' in file:
            sequence = json.load(open(TEST + file))
            testset.append(np.array(sequence))
    testset = np.array(testset)[:,:,1:].transpose(0,2,1) # N,C,L
    N_test = testset.shape[0]

    trainset = []
    for file in os.listdir(TRAIN):
        if '.json' in file:
            sequence = json.load(open(TRAIN + file))
            trainset.append(np.array(sequence))
    trainset = np.array(trainset)[:,:,1:].transpose(0,2,1) # N,C,L
    N_train = trainset.shape[0]

    # train
    cnn.train()
    for epo in range(EPOCH):
        for segment_idx in range(1): #range(DATE_LEN - SEQ_LEN):
            for idx in range(1): #range(N_train // BS):
                # plt.plot(trainset[idx, P_average, segment_idx:segment_idx+SEQ_LEN])
                # plt.show()
                vectors_in  = torch.Tensor(trainset[idx*BS : idx*BS+BS, :, segment_idx : segment_idx+INPUT_LEN])
                vectors_out = torch.Tensor(trainset[idx*BS : idx*BS+BS, :, segment_idx+INPUT_LEN : segment_idx+SEQ_LEN])[:,[P0,Pt]]

                outputs = cnn.forward(vectors_in)

                # evaluate
                loss = lossF(outputs, vectors_out)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                # visualize:
                # if idx == 0:
                plt.plot(vectors_out[0,0,:].detach().numpy())
                plt.plot(outputs[0,0,:].detach().numpy())
                plt.show()

                print("epo = {}, batch = {}, loss = {}\n".format(epo, idx, loss.item()))

    # test
    # cnn.eval()
    # n_up_up = 0
    # n_down_down = 0
    # n_pred_up = 0
    # n_is_up = 0
    # n_pred_down = 0
    # n_is_down = 0
    # for idx in range(N_test):
    #     split = 65
    #     vectors_in  = torch.Tensor(trainset[idx*BS:idx*BS+BS, :split].transpose(1,0,2))
    #     vectors_out = torch.Tensor(trainset[idx*BS:idx*BS+BS, split:].transpose(1,0,2))

    #     outputs = cnn.forward(vectors_in)

    #     loss = lossF(outputs, vectors_out)

    #     trends_pred = __get_trends(outputs[0].detach().numpy())
    #     trends_ans = __get_trends(vectors_out[:,0])
    #     n_up_up = np.sum(trend_preds * trends_ans == 1)     # pred=1, ans=1
    #     n_down_down = np.sum(trend_preds * trends_ans == 4) # pred=2, ans=2
    #     n_pred_up = np.sum(trends_pred == 1)
    #     n_pred_down = np.sum(trends_pred == 2)
    #     n_is_up = np.sum(trends_ans == 1)
    #     n_is_down = np.sum(trends_ans == 2)

    #     print("test: index = {}, loss = {}".format(idx, loss.item()))

    # print("test: P_up = {}, P_down = {}, R_up = {}, R_down = {}".format(\
    #             n_up_up / n_pred_up, n_down_down / n_pred_down, n_up_up / n_is_up, n_down_down / n_is_down))
