import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

import os
import json

import logger
from logger import Logger

import pickle

# training & testing meta-param
EPOCH = 50
BS = 100
DATE_LEN = 14
INPUT_LEN = 10

L1 = 64
L2 = 32

INF = 1e5
EPS = 1e-5
BREAK = 10

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
            nn.Conv1d(FEATURES, 8, kernel_size=3, padding=1),
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.Conv1d(8, 8, kernel_size=1),
        )
        self.bn1 = nn.BatchNorm1d(8)
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16,kernel_size=3, padding=1),
            nn.Conv1d(16,16,kernel_size=3, padding=1),
            nn.Conv1d(16,16,kernel_size=1),
        )
        self.mp2 = nn.Conv1d(16,16,kernel_size=2,stride=2)
        self.bn2 = nn.BatchNorm1d(16)

        self.conv3 = nn.Sequential(
            nn.Conv1d(16,32,kernel_size=3, padding=1),
            nn.Conv1d(32,32,kernel_size=3, padding=1),
            nn.Conv1d(32,32,kernel_size=1),
        )
        self.mp3 = nn.Conv1d(32,32,kernel_size=2,stride=2)
        self.bn3 = nn.BatchNorm1d(32)

        self.lin_encode = nn.Linear(L1, L2)
        self.lin_classify = nn.Linear(L2, 3)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        
        x = self.mp2(F.relu(self.conv2(x)))
        x = self.bn2(x)

        x = self.mp3(F.relu(self.conv3(x)))
        x = self.bn3(x)

        x = x.reshape(-1, L1)
        x = F.relu(self.lin_encode(x))
        x = self.lin_classify(x)

        return x


def __get_trends(batch_vectors):
    # trends = np.array([(v[Pt] - v[P0] / v[P0]) for v in batch_vectors])
    trends = (batch_vectors[:,Pt] - batch_vectors[:,P0]) / (batch_vectors[:,P0] + EPS)
    trends[abs(trends) < THETA] = 0
    trends[trends > 0] = 1
    trends[trends < 0] = 2
    return trends
    # returns a [0/1/2] batch, 0 for ->, 1 for ↑, 2 for ↓

if __name__ == '__main__':
    cnn = CNN_stock()
    optimizer = torch.optim.SGD(cnn.parameters(), momentum=0.9, lr=0.01)
    lossF = nn.CrossEntropyLoss()

    # dataset
    TEST = "./TickJson/test/"
    TRAIN = "./TickJson/train/"
    VALID = "./TickJson/valid/"

    validset = []
    for file in os.listdir(VALID):
        if '.json' in file:
            sequence = json.load(open(VALID + file))
            validset.append(np.array(sequence))
    validset = np.array(validset).transpose(0,2,1) # N,C,L
    N_valid = validset.shape[0]

    trainset = []
    for file in os.listdir(TRAIN):
        if '.json' in file:
            sequence = json.load(open(TRAIN + file))
            trainset.append(np.array(sequence))
    trainset = np.array(trainset).transpose(0,2,1) # N,C,L
    N_train = trainset.shape[0]

    # train
    logger = Logger("./log")
    prev_valid_accu = 0
    break_cnt = 0
    for epo in range(EPOCH):
        # training
        cnn.train()
        train_accu = []
        for segment_idx in range(DATE_LEN - INPUT_LEN - 1):
            for idx in range(N_train // BS):
                vectors_in  = torch.Tensor(trainset[idx*BS : idx*BS+BS, 1:, segment_idx : segment_idx+INPUT_LEN])
                ans = torch.LongTensor(__get_trends(trainset[idx*BS : idx*BS+BS, :, segment_idx+INPUT_LEN+1]))

                outputs = cnn.forward(vectors_in)

                # evaluate
                loss = lossF(outputs, ans)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                corr = np.sum(np.argmax(outputs.detach().numpy(), axis=1) == ans.detach().numpy())
                train_accu.append(corr / BS)

                logger.scalar_summary("loss", loss, (N_train // BS) * (DATE_LEN - INPUT_LEN - 1) * epo + N_train // BS * segment_idx + idx)
                logger.scalar_summary("accu", corr / BS, (N_train // BS) * (DATE_LEN - INPUT_LEN - 1) * epo + N_train // BS * segment_idx + idx)

            print("segment idx = {}".format(segment_idx))

        print("train: epo = {}, train-accu = {}".format(epo, np.mean(train_accu)))

        # validation
        cnn.eval()
        valid_accu = []
        for segment_idx in range(DATE_LEN - INPUT_LEN - 1):
            vectors_in = torch.Tensor(validset[:, 1:, segment_idx : segment_idx+INPUT_LEN])
            ans = torch.LongTensor(__get_trends(validset[:, :, segment_idx+INPUT_LEN+1]))

            outputs = cnn.forward(vectors_in)

            loss = lossF(outputs, ans)
            corr = np.sum(np.argmax(outputs.detach().numpy(), axis=1) == ans.detach().numpy())

            valid_accu.append(corr / validset.shape[0])

        avg_valid_accu = np.mean(valid_accu)
        print("valid: epo = {}, valid-accu = {}".format(epo, avg_valid_accu))
        if avg_valid_accu >= prev_valid_accu:
            torch.save(cnn.state_dict(), "./model_best.pkl")
        # else:
            # break_cnt += 1
            # if break_cnt > BREAK: break

        prev_valid_accu = avg_valid_accu

    torch.save(cnn.state_dict(), "./model.pkl")
