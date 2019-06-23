import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import cnn_stock_classify
from cnn_stock_classify import CNN_stock

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
BREAK = 1

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

def __get_trends(batch_vectors):
    # trends = np.array([(v[Pt] - v[P0] / v[P0]) for v in batch_vectors])
    trends = (batch_vectors[:,Pt] - batch_vectors[:,P0]) / (batch_vectors[:,P0] + EPS)
    trends[abs(trends) < THETA] = 0
    trends[trends > 0] = 1
    trends[trends < 0] = 2
    return trends
    # returns a [0/1/2] batch, 0 for ->, 1 for ↑, 2 for ↓

if __name__ == '__main__':
    # dataset
    TEST = "./TickJson/test/"
    TRAIN = "./TickJson/train/"
    VALID = "./TickJson/valid/"
    testset = []
    for file in os.listdir(TEST):
        if '.json' in file:
            sequence = json.load(open(TEST + file))
            testset.append(np.array(sequence))
    testset = np.array(testset).transpose(0,2,1) # N,C,L
    N_test = testset.shape[0]

    # test
    cnn = CNN_stock()
    # cnn.load_state_dict(torch.load("./model.pkl"))
    cnn.load_state_dict(torch.load("./model_best.pkl"))
    cnn.eval()
    n_up_up = 0
    n_down_down = 0
    n_pred_up = 0
    n_is_up = 0
    n_pred_down = 0
    n_is_down = 0
    for segment_idx in range(DATE_LEN - INPUT_LEN - 1):
        for idx in range(N_test // BS):
            vectors_in = torch.Tensor(testset[idx*BS:idx*BS+BS, 1:, segment_idx : segment_idx+INPUT_LEN])
            ans = __get_trends(testset[idx*BS : idx*BS+BS, :, segment_idx+INPUT_LEN+1])

            outputs = cnn.forward(vectors_in)
            guesses = np.argmax(outputs.detach().numpy(), axis=1)

            n_up_up = np.sum(guesses * ans == 1)     # pred=1, ans=1
            n_down_down = np.sum(guesses * ans == 4) # pred=2, ans=2
            n_pred_up = np.sum(guesses == 1)
            n_pred_down = np.sum(guesses == 2)
            n_is_up = np.sum(ans == 1)
            n_is_down = np.sum(ans == 2)

            corr = np.sum(np.argmax(outputs.detach().numpy(), axis=1) == ans)
            print("test: index = {}, corr = {}".format(idx, corr))

    print("test: P_up = {}, P_down = {}, R_up = {}, R_down = {}".format(\
                n_up_up / n_pred_up, n_down_down / n_pred_down, n_up_up / n_is_up, n_down_down / n_is_down))