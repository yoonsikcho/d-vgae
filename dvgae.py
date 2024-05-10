import os.path as osp
import argparse
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops)
from sklearn.metrics import average_precision_score, roc_auc_score

import argparse
import os.path as osp
import time

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, APPNP, GCNConv

EPS = 1e-15
MAX_LOGSTD = 10
MAX_TEMP = 2.0 
MIN_TEMP = 0.1 

sc = 0.8 

decay_weight = np.log(MAX_TEMP/MIN_TEMP)
decay_step = 150.0
patience = 40

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DVGAE')
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=800)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=transform)
train_data, val_data, test_data = dataset[0]

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index,not_prop=0):

        if args.model == 'DVGAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)

            x = self.linear2(x)
            x = F.normalize(x,p=2,dim=1) * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x, x_
        
class Encoder2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(Encoder2, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index,not_prop=0):
        
        if args.model == 'DVGAE':
            x_ = self.linear1(x)

            x = self.linear2(x)
            x = F.normalize(x,p=2,dim=1) * sc
            return x, x_ 
        
class DVGAE(torch.nn.Module):
    def __init__(self, encoder1, encoder2, decoder):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        DVGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder1)
        reset(self.encoder2)
        reset(self.decoder)
  
    def encode1(self, *args, **kwargs):
        """"""
        self.__mu1__, self.__logstd1__ = self.encoder1(*args, **kwargs)
        self.__logstd1__ = self.__logstd1__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu1__, self.__logstd1__)
        return z
    def encode2(self, *args, **kwargs):
        """"""
        self.__mu2__, self.__logstd2__ = self.encoder2(*args, **kwargs)
        self.__logstd2__ = self.__logstd2__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu2__, self.__logstd2__)
        return z
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu 
    
    def test(self, z1, z2, temp, pos_edge_index, neg_edge_index):

        pos_y = z1.new_ones(pos_edge_index.size(1))
        neg_y = z1.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        
        pos_pred = self.decoder(z1, z2, temp, pos_edge_index, sigmoid=True, training=False)
        neg_pred = self.decoder(z1, z2, temp, neg_edge_index, sigmoid=True, training=False)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
  
    def recon_loss(self, z1, z2, temp, pos_edge_index, neg_edge_index=None):

        decode_p = self.decoder(z1, z2, temp, pos_edge_index, sigmoid=True, training=True)
        pos_loss = -torch.log(decode_p + EPS).sum()

        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z1.size(0))
        
        decode_n1 = self.decoder(z1, z2, temp, neg_edge_index, sigmoid=True, training=True)
        neg_loss = -torch.log(1 -decode_n1 + EPS).sum() 

        return (pos_loss + neg_loss) 
    
    def kl_loss1(self, mu=None, logstd=None):

        mu = self.__mu1__ if mu is None else mu
        logstd = self.__logstd1__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        
    def kl_loss2(self, mu=None, logstd=None):

        mu = self.__mu2__ if mu is None else mu
        logstd = self.__logstd2__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    
class InnerProductDecoder2(torch.nn.Module):
    def __init__(self):
        super().__init__()        

    def forward(self, z1, z2,  temp, edge_index, sigmoid=True, training=True):
            
        if training: 
            z11 = z1.detach().clone()
            vf = (z11[edge_index[0]] * z11[edge_index[1]]).sum(dim=1) 
            la = torch.cat(  (torch.unsqueeze(vf, 1), torch.zeros(torch.unsqueeze(vf, 1).shape).to(device)   ),1)
            la_ra = la
            a = F.gumbel_softmax((la_ra), tau=temp, hard=True)[:,:1]
            value_feature = (z1[edge_index[0]] * z1[edge_index[1]]).sum(dim=1)
            value_network =  z2[edge_index[0],[0]] + z2[edge_index[1],[0]]
            feature_flag = torch.flatten(a)
            return feature_flag*torch.sigmoid(value_feature) + (1-feature_flag)*torch.sigmoid(value_network) if sigmoid else value
            
        else:
            z11 = z1.detach().clone()
            vf = (z11[edge_index[0]] * z11[edge_index[1]]).sum(dim=1)
            la = torch.cat(  (torch.unsqueeze(vf, 1), torch.zeros(torch.unsqueeze(vf, 1).shape).to(device)   ),1)
            la_ra = la
            a = F.softmax((la_ra), dim=1)[:,:1]
            value_feature = (z1[edge_index[0]] * z1[edge_index[1]]).sum(dim=1)
            value_network =  z2[edge_index[0],[0]] + z2[edge_index[1],[0]]         
            return torch.sigmoid(value_feature)*torch.sigmoid(value_feature) + (1-torch.sigmoid(value_feature))*torch.sigmoid(value_network) if sigmoid else value

    def forward_all(self, z, sigmoid=True):

        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

channels = args.channels    
in_channels= dataset.num_features
N = train_data.x.shape[0]

model = DVGAE(Encoder(in_channels, channels, train_data.edge_index), Encoder2(N, 2, train_data.edge_index) , InnerProductDecoder2()).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

network_input = torch.eye(N).to(device)
l1 = train_data.edge_index

def train(epoch):
    temp = np.maximum(MAX_TEMP*np.exp(-(epoch-1)/decay_step*decay_weight), MIN_TEMP)

    model.train()
    optimizer.zero_grad()
    z1 = model.encode1( train_data.x , l1)
    z2 = model.encode2(network_input, l1)
    loss = model.recon_loss(z1,z2, temp, train_data.pos_edge_label_index)

    loss = loss + (1.0 / N) * (model.kl_loss1()+model.kl_loss2())
    loss.backward()
    optimizer.step()
    return loss

def test(pos_edge_index, neg_edge_index, selected_list, plot_his=0):
    model.eval()
    with torch.no_grad():
        z1 = model.encode1(train_data.x, selected_list)
        z2 = model.encode2(network_input, selected_list)
    return model.test(z1, z2, 1.0, pos_edge_index, neg_edge_index)

early_stopping = EarlyStopping(patience = patience, verbose = True)

for epoch in range(1,args.epochs + 1):
    loss  = train(epoch)
    loss = float(loss)
    
    with torch.no_grad():
        val_pos, val_neg = val_data.pos_edge_label_index, val_data.neg_edge_label_index
        auc, ap = test(val_pos, val_neg, train_data.edge_index)
        if epoch>200: # after minimum epoch, check early stopping
            early_stopping(-auc, model)
            if early_stopping.early_stop:
                break

model.load_state_dict(torch.load('checkpoint.pt'))

test_pos, test_neg = test_data.pos_edge_label_index, test_data.neg_edge_label_index
auc, ap = test(test_pos, test_neg,train_data.edge_index)

print('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f} AP: {:.4f}'.format(epoch, loss, auc, ap))
