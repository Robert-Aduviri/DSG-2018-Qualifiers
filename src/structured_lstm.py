import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from src.neuralnet import NeuralNet
from src.utils import to_cat_codes, apply_cats

def scale_features(df, scaler, num_cols):
    scaled = scaler.transform(df[num_cols])
    for i, col in enumerate(num_cols):
        df[col] = scaled[:,i]

def preprocess(train, val, test, cat_cols, num_cols, seq_interest,
               seq_transactions, seq_buysells, seq_customers, seq_isins):
    print('Encoding cats...')
    to_cat_codes(train, cat_cols)
    apply_cats(val, train)
    apply_cats(test, train)

    for col in cat_cols:
        train[col] = train[col].cat.codes
        val[col] = val[col].cat.codes
        test[col] = test[col].cat.codes
    
    # Fix nan cats
    nan_cols = [c for c in cat_cols if \
            any(df[c].min() < 0 for df in [train, val, test])]
    for c in nan_cols:
        train[c] = train[c] + 1
        val[c] = val[c] + 1
        test[c] = test[c] + 1
        
    print('Scaling conts...')
    scaler = StandardScaler().fit(pd.concat([train[num_cols], 
                                  val[num_cols], test[num_cols]]))

    scale_features(train, scaler, num_cols)
    scale_features(val, scaler, num_cols)
    scale_features(test, scaler, num_cols)
    
    
    print('Extracting seqs...')
    train_seqs = np.array([[seq_interest[(c,i,b)],
                            seq_transactions[(c,i,b)],
                            seq_buysells[(c,i)],
                            seq_customers[c],
                            seq_isins[i]] for c,i,b in \
                zip(train.CustomerIdx, train.IsinIdx, train.BuySell)])
    val_seqs = np.array([[seq_interest[(c,i,b)],
                            seq_transactions[(c,i,b)],
                            seq_buysells[(c,i)],
                            seq_customers[c],
                            seq_isins[i]] for c,i,b in \
                    zip(val.CustomerIdx, val.IsinIdx, val.BuySell)])
    test_seqs = np.array([[seq_interest[(c,i,b)],
                            seq_transactions[(c,i,b)],
                            seq_buysells[(c,i)],
                            seq_customers[c],
                            seq_isins[i]] for c,i,b in \
                    zip(test.CustomerIdx, test.IsinIdx, test.BuySell)])
    
    train['BuySell'] = train.BuySell.apply(lambda x: int(x == 'Buy'))
    val['BuySell'] = val.BuySell.apply(lambda x: int(x == 'Buy'))
    test['BuySell'] = test.BuySell.apply(lambda x: int(x == 'Buy'))
    num_cols.append('BuySell')
    
    return scaler, train_seqs, val_seqs, test_seqs
    
class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, cats, conts, seqs, targets=None):
        self.cats = cats.values.astype(np.int64)
        self.conts = conts.values.astype(np.float32)
        self.seqs = np.array(seqs).astype(np.float32)
        self.targets = np.array(targets).astype(np.float32) \
                            if targets is not None else \
                            np.zeros(len(cats)).astype(np.float32)
    
    def __len__(self):
        return len(self.cats)
    
    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx],
                self.seqs[idx], self.targets[idx]]
    
class MultimodalNet(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, szs, drops, 
                 rnn_hidden_sz, rnn_input_sz, rnn_n_layers, rnn_drop,
                 out_sz=1):
        super().__init__()
        self.structured_net = NeuralNet(emb_szs, n_cont=n_cont, 
                        emb_drop=emb_drop, szs=szs, drops=drops, 
                        out_sz=rnn_hidden_sz)
        
        self.lstm = nn.LSTM(rnn_input_sz, rnn_hidden_sz, rnn_n_layers, 
                            dropout=rnn_drop)
        self.out = nn.Linear(rnn_hidden_sz * 2, out_sz) # [struct_out, rnn_out]
        
        self.rnn_n_layers = rnn_n_layers
        self.rnn_hidden_sz = rnn_hidden_sz
        
    def forward(self, cats, conts, seqs, hidden):
        # seqs: [bs, inp, seq]
        x = self.structured_net(cats, conts) # [bs, hs]
        # cell = x.unsqueeze(0).repeat(self.rnn_n_layers, 1, 1) # [nlay, bs, hs]
        cell = x.unsqueeze(0).expand(self.rnn_n_layers, -1, -1).contiguous()
        seqs = seqs.transpose(1,0).transpose(2,0) # .unsqueeze(2) 
        # seqs: [seq, bs, inp]
        outputs, hidden = self.lstm(seqs, (hidden, cell))
        out = self.out(torch.cat([x, outputs[-1]], 1)) # [struct_out, rnn_out]
        return out
        
    def init_hidden(self, batch_sz):
        return torch.zeros(self.rnn_n_layers, batch_sz, self.rnn_hidden_sz)
    
def train_step(model, cats, conts, seqs, hidden, 
               targets, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    preds = model(cats, conts, seqs, hidden)
    loss = criterion(preds.view(-1), targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def get_predictions(model, data_loader, print_every=1200, USE_CUDA=False):
    targets = []
    preds = []
    model.eval()
    for batch_idx, (cats, conts, seqs, target) in enumerate(data_loader):
        with torch.no_grad():            
            hidden = model.init_hidden(len(cats))
            if USE_CUDA:
                cats, conts, seqs, target, hidden = cats.cuda(), conts.cuda(), \
                                    seqs.cuda(), target.cuda(), hidden.cuda()
            pred = model(cats, conts, seqs, hidden)
            targets.extend(target.cpu())
            preds.extend(pred.cpu())
            assert len(targets) == len(preds)
            if batch_idx % print_every == 0:
                print('[{}/{} ({:.0f}%)]'.format(
                        batch_idx * len(cats), len(data_loader.dataset),
                        100. * batch_idx / len(data_loader)))
    return [x.item() for x in targets], [F.sigmoid(x).item() for x in preds]

def get_metrics(model, data_loader, USE_CUDA=False):
    targets, preds = get_predictions(model, data_loader, USE_CUDA=USE_CUDA)
    loss = nn.BCELoss()(torch.Tensor(preds), torch.Tensor(targets)).item()
    auc = roc_auc_score(targets, preds)
    return loss, auc
    
def train_model(model, train_loader, val_loader, optimizer, criterion,
                n_epochs, print_every=200, val_every=5, USE_CUDA=False):
    if USE_CUDA:
        model = model.cuda()
    train_losses = []
    val_losses = []
    val_auc_scores = []
    val_every *= print_every
    for epoch in range(n_epochs):
        train_loss = 0
        for batch_idx, (cats, conts, seqs, target) in enumerate(train_loader):
            hidden = model.init_hidden(len(cats))
            if USE_CUDA:
                cats, conts, seqs, target, hidden = cats.cuda(), conts.cuda(), \
                                    seqs.cuda(), target.cuda(), hidden.cuda()
            train_loss += train_step(model, cats, conts, seqs, hidden, 
                                     target, optimizer, criterion)
            
            if batch_idx > 0 and batch_idx % print_every == 0:
                train_loss /= print_every
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(seqs), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), train_loss))
                train_losses.append(train_loss)
                train_loss = 0
            
            if val_loader is not None and batch_idx > 0 and batch_idx % val_every == 0:
                val_loss, val_auc = get_metrics(model, val_loader, USE_CUDA)
                val_losses.append(val_loss)
                val_auc_scores.append(val_auc)
                print(f'ROC AUC Score: {val_auc:.6f}') 
                print(f'Validation Loss: {val_loss:.6f}')
        
        if val_loader is not None:
            print('Epoch Results:')
            train_loss, train_auc = get_metrics(model, train_loader, USE_CUDA)
            print(f'Train ROC AUC Score: {train_auc:.6f}')
            print(f'Train Loss: {train_loss:.6f}')
            val_loss, val_auc = get_metrics(model, val_loader, USE_CUDA)
            print(f'Validation ROC AUC Score: {val_auc:.6f}')
            print(f'Validation Loss: {val_loss:.6f}')       
        
        print()
    return model, train_losses, val_losses, val_auc_scores   