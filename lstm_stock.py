
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

import os
import json

EPOCH = 300
BS = 1
THETA = .004
TOTAL_LEN = 3

INPUT_DIM = 7

TIMESTAMP = 0
P0 = 1
P_low = 2
P_high = 3
Pt = 4
P_average = 5
VOL = 6
TURNOVER = 7

class LSTM_stock(nn.Module):

    def __init__(self, input_dim, cell_dim):
        super(LSTM_stock, self).__init__()
        self.lstm_block = nn.LSTM(input_dim, cell_dim, 2)

    def forward(self, seq):
        # seq: L x B x D
        # L = split
        # B = 5
        # D = 8
        encode_outputs, (h, c) = self.lstm_block(seq)
        self.h = h
        self.h.retain_grad()
        self.c = c
        self.c.retain_grad()
        self.encode_outputs = encode_outputs[-1]
        self.encode_outputs.retain_grad()

        outputs = []
        decoder_x = self.encode_outputs.view(1, -1, INPUT_DIM)

        for idx in range(TOTAL_LEN - seq.shape[0]):
            outputs.append(decoder_x)
            decoder_x, (h, c) = self.lstm_block(decoder_x, (h, c))
            # decoder_x, _ = self.lstm_block(decoder_x)

        self.output = torch.cat(outputs, dim=0)
        self.output.retain_grad()
        return self.output


def __get_trends(vector_batch):
    trends = [(v[Pt] - v[P0] / v[P0]) for v in batch]
    trends[abs(trends) < THETA] = 0
    trends[trends > 0] = 1
    trends[trends < 0] = 2
    return trends
    # returns a [0/1/2] batch, 0 for ->, 1 for ↑, 2 for ↓


if __name__ == '__main__':
    lstm = LSTM_stock(input_dim=INPUT_DIM, cell_dim=INPUT_DIM)
    optimizer = torch.optim.SGD(lstm.parameters(), lr=1)
    lossF = nn.MSELoss()

    # dataset
    TEST = "./TickJson/test/"
    TRAIN = "./TickJson/train/"
    testset = []
    for file in os.listdir(TEST):
        if '.json' in file:
            sequence = json.load(open(TEST + file))
            testset.append(np.array(sequence))
    testset = np.array(testset)[:,:,1:]
    N_test = testset.shape[0]

    trainset = []
    for file in os.listdir(TRAIN):
        if '.json' in file:
            sequence = json.load(open(TRAIN + file))
            trainset.append(np.array(sequence))
    trainset = np.array(trainset)[:,:,1:]
    N_train = trainset.shape[0]

    def get_weight0(lstm):
        cnt = 0
        for weight in lstm.lstm_block.parameters():
            if cnt < 1:
                cnt += 1
                continue
            return weight

    # train
    lstm.train()
    for epo in range(EPOCH):
        for idx in range(1): #(N_train // BS):
            # split = np.random.choice(np.arange(55,65))
            # split = int(TOTAL_LEN // 2)
            split = 1
            vectors_in  = torch.Tensor(trainset[idx*BS:idx*BS+BS, :split].transpose(1,0,2))
            vectors_in.requires_grad = True
            vectors_out = torch.Tensor(trainset[idx*BS:idx*BS+BS, split:TOTAL_LEN].transpose(1,0,2))

            outputs = lstm.forward(vectors_in)

            # evaluate
            print("outputs\n", torch.mean(outputs, dim=(1,2)), "\nanswer\n", torc.mean(vectors_out, dim=(1,2)))
            loss = lossF(outputs, vectors_out)
            loss.backward()

            optimizer.step()

            # grads
            # print("grad = ", torch.sum(get_weight0(lstm).grad) * 1e10)
            print("seq grad = ", torch.mean(vectors_in.grad))
            # print("h grad = ", np.mean(lstm.h.grad.detach().numpy()))
            # print("c grad = ", np.mean(lstm.c.grad.detach().numpy()))
            print("encode_outputs grad = ", np.mean(lstm.encode_outputs.grad.detach().numpy()))
            print("output grad = ", np.mean(lstm.output.grad.detach().numpy(), axis=(1,2)))

            optimizer.zero_grad()

            # visualize:
            plt.plot(vectors_out[:,0,P_average].detach().numpy())
            plt.plot(outputs[:,0,P_average].detach().numpy())
            plt.show()

            print("epo = {}, batch = {}, loss = {}\n".format(epo, idx, loss.item()))

    # test
    lstm.eval()
    n_up_up = 0
    n_down_down = 0
    n_pred_up = 0
    n_is_up = 0
    n_pred_down = 0
    n_is_down = 0
    for idx in range(N_test):
        split = 65
        vectors_in  = torch.Tensor(trainset[idx*BS:idx*BS+BS, :split].transpose(1,0,2))
        vectors_out = torch.Tensor(trainset[idx*BS:idx*BS+BS, split:].transpose(1,0,2))

        outputs = lstm.forward(vectors_in)

        loss = lossF(outputs, vectors_out)

        trends_pred = __get_trends(outputs[0].detach().numpy())
        trends_ans = __get_trends(vectors_out[:,0])
        n_up_up = np.sum(trend_preds * trends_ans == 1)     # pred=1, ans=1
        n_down_down = np.sum(trend_preds * trends_ans == 4) # pred=2, ans=2
        n_pred_up = np.sum(trends_pred == 1)
        n_pred_down = np.sum(trends_pred == 2)
        n_is_up = np.sum(trends_ans == 1)
        n_is_down = np.sum(trends_ans == 2)

        print("test: index = {}, loss = {}".format(idx, loss.item()))

    print("test: P_up = {}, P_down = {}, R_up = {}, R_down = {}".format(\
                n_up_up / n_pred_up, n_down_down / n_pred_down, n_up_up / n_is_up, n_down_down / n_is_down))
