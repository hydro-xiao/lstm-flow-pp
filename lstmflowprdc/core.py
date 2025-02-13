import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Funct
from torch.utils.data import Dataset as TorchData

############ CLASS: loss function section
class RMSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            temp = torch.sqrt(((p0 - t0) ** 2).mean())
            loss = loss + temp
        return loss

############ CLASS: dataset section
class seqDataset(TorchData):
    def __init__(self, dataset, target, features, seq_len=12):
        self.features = features
        self.target = target
        self.seq_len = seq_len
        self.yt = torch.from_numpy(np.array(dataset[target].to_array()))
        self.y  = self.yt.permute(2,1,0)
        self.Xt = torch.from_numpy(np.array(dataset[features].to_array()))
        self.X  = self.Xt.permute(2,1,0)
        self.X  = self.Xt.permute(2,1,0)
        self.step = self.X.shape[0]                ## total time steps
        self.lps  = 1+ int( np.ceil((self.step-self.seq_len)/12.) ) ## ttl loops

    def __len__(self):
        ttlstep = self.X.shape[0]                ## total time steps
        ttlps   = 1+ int( np.ceil((ttlstep-self.seq_len)/12.) ) ## ttl loops
        return int(ttlps)

    def __getitem__(self, i): 
        if i==self.lps-1:   ### deal with last batch
            x = self.X[self.step-self.seq_len:, :, :]
            y = self.y[self.step-self.seq_len:, :, :]
        else:     ### normal case
            istart = i*12
            ind    = istart+self.seq_len
            x = self.X[istart:ind, :, :]
            y = self.y[istart:ind, :, :]
        return x, y

    def shape(self):
        print(self.X.shape)
        print(self.y.shape)
        print(self.X.shape[0], self.seq_len)
        print(self.X.shape[0]-self.seq_len)
        print(self.step)
        print(self.lps)


############ CLASS: nerual network section
class LSTMmodel(nn.Module):
    def __init__(self, *, nx, ny, hiddensize, dr=0.5):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hiddensize = hiddensize
        self.num_layers = 1
        self.linearIn = nn.Linear(nx, hiddensize)
        self.lstm = nn.LSTM(input_size=hiddensize,\
                hidden_size=hiddensize, num_layers=self.num_layers)
        self.linearOut = nn.Linear(hiddensize, ny)

    def forward(self, x, doDropMC=False):
        batch_size = x.shape[1]
        ntstep     = x.shape[0]
        output     = torch.zeros(ntstep, batch_size, self.ny)
        for ti in range(ntstep):
            h0 = torch.zeros(1, batch_size, \
                self.hiddensize).requires_grad_()
            c0 = torch.zeros(1, batch_size, \
                self.hiddensize).requires_grad_()
            #h0 = None
            #c0 = None

            xt = torch.unsqueeze(x[ti,:,:],0)
            x0 = Funct.relu(self.linearIn(xt.to(torch.float32)))
            x1, (h1, c1) = self.lstm(x0, (h0, c0) )
            yi  = self.linearOut(h1)
            output[ti,:,:] = yi
        return output


