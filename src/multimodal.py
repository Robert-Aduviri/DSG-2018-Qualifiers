import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import roc_auc_score

from src.neuralnet import NeuralNet
from src.lstm import LSTMClassifier

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, cats, conts, seqs, targets=None):
        self.cats = cats.values.astype(np.int64)
        self.conts = conts.values.astype(np.float32)
        self.seqs = np.array(seqs).astype(np.float32)
        self.targets = np.array(targets).astype(np.float32) \
                            if targets is not None else \
                            np.zeros_like(seqs).astype(np.float32)
    
    def __len__(self):
        return len(self.cats)
    
    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx],
                self.seqs[idx], self.targets[idx]]
    
class MultimodalClassifier(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, szs, drops, 
                 rnn_hidden_sz, rnn_input_sz, rnn_n_layers, rnn_drop):
        super().__init__()
        self.structured_net = NeuralNet(emb_szs, n_cont=n_cont, 
                        emb_drop=emb_drop, szs=szs, drops=drops, 
                        out_sz=rnn_hidden_sz * rnn_n_layers * 2)
        
        self.sequential_net = LSTMClassifier(input_sz=rnn_input_sz,
                        hidden_sz=rnn_hidden_sz, n_layers=rnn_n_layers, 
                        drop=rnn_drop)  
        self.rnn_n_layers = rnn_n_layers
        self.rnn_hidden_sz = rnn_hidden_sz
        
    def forward(self, cats, conts):
        out = self.structured_net(cats, conts)
        out = out.view(-1, 2, self.rnn_n_layers, self.rnn_hidden_sz) \
                    .transpose(0,1).transpose(1,2)
        return (out[0].contiguous(), out[1].contiguous())
        
def train_step(model, cats, conts, seqs, targets, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    hidden = model(cats, conts)
    seqs = seqs.transpose(0,1) # [seq_len, batch_sz]
    targets = targets.transpose(0,1)
    loss = 0
    for i in range(len(seqs)): # for each timestep
        output, hidden = model.sequential_net(seqs[i].unsqueeze(0) \
                                              .unsqueeze(2), hidden)
        loss += criterion(output, targets[i].unsqueeze(1))
    loss.backward()
    optimizer.step()
    return loss.item() / len(seqs)

def evaluate(model, cats, conts, seqs):
    with torch.no_grad():
        model.eval()
        hidden = model(cats, conts)
        seqs = seqs.transpose(0,1) # [seq_len, batch_sz]
        for i in range(len(seqs)): # for each timestep
            output, hidden = model.sequential_net(seqs[i].unsqueeze(0) \
                                                  .unsqueeze(2), hidden)
        return F.sigmoid(output).view(-1)
    
def get_predictions(model, data_loader, print_every=800, USE_CUDA=False):
    all_targets = []
    all_preds = []
    for batch_idx, (cats, conts, seqs, targets) in enumerate(data_loader):
        with torch.no_grad():
            if USE_CUDA:
                cats, conts, seqs, targets = cats.cuda(), conts.cuda(), \
                                             seqs.cuda(), targets.cuda()
            preds = evaluate(model, cats, conts, seqs)
            all_targets.extend(targets.cpu().numpy()[:,-1]) # last timestemp
            all_preds.extend(preds.cpu().numpy())
            assert len(all_targets) == len(all_preds)
            if batch_idx % print_every == 0:
                print('[{}/{} ({:.0f}%)]'.format(
                        batch_idx * len(seqs), len(data_loader.dataset),
                        100. * batch_idx / len(data_loader)))
    return all_targets, all_preds

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
        for batch_idx, (cats, conts, seqs, targets) in enumerate(train_loader):
            if USE_CUDA:
                cats, conts, seqs, targets = cats.cuda(), conts.cuda(), \
                                             seqs.cuda(), targets.cuda()
            train_loss += train_step(model, cats, conts, seqs, targets, 
                                     optimizer, criterion)
            
            if batch_idx > 0 and batch_idx % print_every == 0:
                train_loss /= print_every
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(seqs), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), train_loss))
                train_losses.append(train_loss)
                train_loss = 0
            
            if batch_idx > 0 and batch_idx % val_every == 0:
                targets, preds = get_predictions(model, val_loader, USE_CUDA=USE_CUDA)
                val_loss = nn.BCELoss()(torch.Tensor(preds),
                                        torch.Tensor(targets)).item()
                val_losses.append(val_loss)
                val_auc = roc_auc_score(targets, preds)
                val_auc_scores.append(val_auc)
                print(f'ROC AUC Score: {val_auc:.6f}') 
                print(f'Validation Loss: {val_loss:.6f}')
        print()
    return model, train_losses, val_losses, val_auc_scores            