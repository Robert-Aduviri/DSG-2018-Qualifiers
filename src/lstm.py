import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from sklearn.metrics import roc_auc_score

class SequentialDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, targets):
        self.seqs = np.array(seqs).astype(np.float32)
        self.targets = np.array(targets).astype(np.float32)
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        return [self.seqs[idx], self.targets[idx]]
    
# params: (input_size, hidden_size, num_layers, bias, 
#          batch_first, dropout, bidirectional)
#   input (seq_len, batch, input_size)
#   (h_0, c_0) (num_layers * num_directions, batch, hidden_size)
#   output (seq_len, batch, hidden_size * num_directions)

class LSTMClassifier(nn.Module):
    def __init__(self, input_sz, hidden_sz, n_layers, drop=0.1, 
                 USE_CUDA=False):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.n_layers = n_layers
        self.USE_CUDA = USE_CUDA
        
        self.lstm = nn.LSTM(input_sz, hidden_sz, n_layers, 
                            batch_first=False, dropout=drop)
        self.out = nn.Linear(hidden_sz, 1) # output_sz 1
        
    def forward(self, sequence, hidden):
        out, hidden = self.lstm(sequence, hidden)
        out = self.out(out[-1])
        return out, hidden
        
    def init_hidden(self, batch_sz):
        h0 = torch.zeros(self.n_layers, batch_sz, self.hidden_sz)
        c0 = torch.zeros(self.n_layers, batch_sz, self.hidden_sz)
        if self.USE_CUDA:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)
    
def train_step(model, seqs, targets, optimizer, criterion):
    '''
    seqs: (batch_sz, seq_len)
    targets: (batch_sz, seq_len)
    '''
    model.train()
    optimizer.zero_grad()
    hidden = model.init_hidden(len(seqs)) # len(seqs) == batch_sz
    seqs = seqs.transpose(0,1) # [seq_len, batch_sz]
    targets = targets.transpose(0,1)
    loss = 0
    for i in range(len(seqs)): # for each timestep
        output, hidden = model(seqs[i].unsqueeze(0).unsqueeze(2), hidden)
        loss += criterion(output, targets[i].unsqueeze(1))
    loss.backward()
    optimizer.step()
    return loss.item() / len(seqs)

def evaluate(model, seqs, criterion=None, targets=None):
    '''
    seqs: (batch_sz, seq_len)
    targets: (batch_sz, seq_len)
    '''
    with torch.no_grad():
        model.eval()
        hidden = model.init_hidden(len(seqs)) # len(seqs) == batch_sz
        seqs = seqs.transpose(0,1) # [seq_len, batch_sz]
        if targets is not None:
            targets = targets.transpose(0,1)
            loss = 0
        output = None
        for i in range(len(seqs)): # for each timestep
            output, hidden = model(seqs[i].unsqueeze(0).unsqueeze(2), hidden)
            if targets is not None and criterion is not None:
                loss += criterion(output, targets[i].unsqueeze(1))
        return loss.item() / len(seqs) if targets is not None else None, \
               F.sigmoid(output).view(-1)

def get_predictions(model, data_loader, criterion, print_every=800, USE_CUDA=False):
    all_targets = []
    all_preds = []
    for batch_idx, (seqs, targets) in enumerate(data_loader):
        with torch.no_grad():
            if USE_CUDA:
                seqs, targets = seqs.cuda(), targets.cuda()
            loss, preds = evaluate(model, seqs, criterion, targets)
            all_targets.extend(targets.cpu().numpy()[:,-1]) # last timestemp
            all_preds.extend(preds.cpu().numpy())
            assert len(all_targets) == len(all_preds)
            if batch_idx % print_every == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_idx * len(seqs), len(data_loader.dataset),
                        100. * batch_idx / len(data_loader), loss))
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
        train_loss, val_loss = 0, 0
        for batch_idx, (seqs, targets) in enumerate(train_loader):
            if USE_CUDA:
                seqs, targets = seqs.cuda(), targets.cuda()
            train_loss += train_step(model, seqs, targets, optimizer, 
                                     criterion)
            
            if batch_idx > 0 and batch_idx % print_every == 0:
                train_loss /= print_every
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(seqs), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), train_loss))
                train_losses.append(train_loss)
                train_loss = 0     
                
            if val_loader is not None and batch_idx > 0 and batch_idx % val_every == 0:
                targets, preds = get_predictions(model, val_loader, criterion, 
                                                 USE_CUDA=USE_CUDA)
                val_loss = nn.BCELoss()(torch.Tensor(preds),
                                        torch.Tensor(targets)).item()
                val_losses.append(val_loss)
                val_auc = roc_auc_score(targets, preds)
                val_auc_scores.append(val_auc)
                print(f'ROC AUC Score: {val_auc:.6f}') 
                print(f'Validation Loss: {val_loss:.6f}')
                
        print()
    return model, train_losses, val_losses, val_auc_scores