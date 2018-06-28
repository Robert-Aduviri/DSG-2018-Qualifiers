import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from sklearn.metrics import roc_auc_score

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, df, cat_cols, num_cols, target_col=None):
        self.cats = df[cat_cols].values.astype(np.int64)
        self.conts = df[num_cols].values.astype(np.float32)
        self.target = df[target_col].values.astype(np.float32) if target_col \
                            else np.zeros((len(df),1)).astype(np.float32)
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.target[idx]]

class NeuralNet(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, szs, drops,
                 use_bn=True, out_sz=1):
        super().__init__()
        
        self.embs = nn.ModuleList([
            nn.Embedding(c, s) for c,s in emb_szs
        ])
        for emb in self.embs:
            self.emb_init(emb)
            
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont = n_emb, n_cont
        szs = [n_emb + n_cont] + szs
        
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)
        ])
        for o in self.lins: 
            nn.init.kaiming_normal_(o.weight.data)
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]
        ])        
            
        self.outp = nn.Linear(szs[-1], out_sz) # 1 output
        nn.init.kaiming_normal_(self.outp.weight.data)
        
        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([
            nn.Dropout(drop) for drop in drops
        ])
        self.bn = nn.BatchNorm1d(n_cont)
        
        self.use_bn = use_bn
    
    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [emb(x_cat[:,i]) for i,emb in enumerate(self.embs)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x2 = self.bn(x_cont)
            x = torch.cat([x, x2], 1) if self.n_emb != 0 else x2
        for lin, drop, bn in zip(self.lins, self.drops, self.bns):
            x = F.relu(lin(x))
            if self.use_bn:
                x = bn(x)
            x = drop(x)
        return self.outp(x) # coupled with BCEWithLogitsLoss
    
    def emb_init(self, x):
        # higher init range for low-dimensional embeddings
        x = x.weight.data
        sc = 2 / (x.size(1) + 1)
        x.uniform_(-sc, sc)

def train_step(cats, conts, target, model, optimizer, criterion, train=True):
    model.train()
    if train:
        optimizer.zero_grad()
    pred = model(cats, conts)
    loss = criterion(pred.view(-1), target) # [preds, targets]
    if train:
        loss.backward()
        optimizer.step()
    return loss.item()

def get_predictions(model, data_loader, print_every=800, USE_CUDA=False):
    all_targets = []
    all_preds = []
    model.eval()
    for batch_idx, (cats, conts, target) in enumerate(data_loader):
        with torch.no_grad():            
            cats, conts, target = Variable(cats), Variable(conts), Variable(target)
            if USE_CUDA:
                cats, conts, target = cats.cuda(), conts.cuda(), target.cuda()
            preds = model(cats, conts)
            all_targets.extend(target.cpu())
            all_preds.extend(preds.cpu())
            if batch_idx % print_every == 0:
                print('[{}/{} ({:.0f}%)]'.format(
                        batch_idx * len(cats), len(data_loader.dataset),
                        100. * batch_idx / len(data_loader)))
    return [x.item() for x in all_targets], [F.sigmoid(x).item() for x in all_preds]

def train_model(model, optimizer, criterion, train_loader, val_loader, 
                n_epochs, print_every=800, val_every=10, USE_CUDA=False):
    if USE_CUDA:
        model = model.cuda()
    train_losses = []
    val_losses = []
    val_auc_scores = []
    val_every *= print_every
    for epoch in range(n_epochs):
        train_loss, val_loss = 0, 0
        for batch_idx, (cats, conts, target) in enumerate(train_loader):
            cats, conts, target = Variable(cats), Variable(conts), Variable(target)
            if USE_CUDA:
                cats, conts, target = cats.cuda(), conts.cuda(), target.cuda()
            train_loss += train_step(cats, conts, target, model, optimizer, 
                                     criterion, train=True)
            if batch_idx > 0 and batch_idx % print_every == 0:
                train_loss /= print_every
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(cats), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), train_loss))
                train_losses.append(train_loss)
                train_loss = 0
                
            if val_loader is not None and batch_idx > 0 and batch_idx % val_every == 0:
                targets, preds = get_predictions(model, val_loader, USE_CUDA=USE_CUDA)
                # [preds, targets]
                val_loss = nn.BCELoss()(torch.Tensor(preds), torch.Tensor(targets)).item() 
                val_losses.append(val_loss)
                val_auc = roc_auc_score(targets, preds)
                val_auc_scores.append(val_auc)
                # [targets, preds]
                print(f'ROC AUC Score: {val_auc:.6f}') 
                print(f'Validation Loss: {val_loss:.6f}')
                
        print()
                
    return model, train_losses, val_losses, val_auc_scores