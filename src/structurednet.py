import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from sklearn.metrics import roc_auc_score

from src.utils import to_cat_codes, apply_cats, week_num

def get_seqs(trade, challenge, week_labels, keys, agg='sum'):
    trade = trade[trade.TradeDateKey >= week_labels[0]].copy()
    if 'Week' not in trade.columns:
        trade['Week'] = trade.TradeDateKey.apply(
                            lambda x: week_num(week_labels, x))
    weeks = trade.groupby(keys + ['Week'], as_index=False) \
                            ['CustomerInterest'].agg(agg)
    n_weeks = weeks.Week.nunique()
    seq_dict = {}
    
    if 'BuySell' in keys:
        new_keys = [k for k in keys if k!='BuySell']
        df = weeks.drop_duplicates(new_keys)
        for tup in zip(*[df[c] for c in new_keys]):
            for b in ['Buy', 'Sell']:
                seq_dict[tup + (b,)] = [0] * n_weeks
        df = challenge.drop_duplicates(new_keys)
        for tup in zip(*[df[c] for c in new_keys]):
            for b in ['Buy', 'Sell']:
                seq_dict[tup + (b,)] = [0] * n_weeks
    else:
        df = weeks.drop_duplicates(keys)
        for tup in zip(*[df[c] for c in keys]):
            tup = tup[0] if len(tup)==1 else tup
            seq_dict[tup] = [0] * n_weeks
        df = challenge.drop_duplicates(keys)
        for tup in zip(*[df[c] for c in keys]):
            tup = tup[0] if len(tup)==1 else tup
            seq_dict[tup] = [0] * n_weeks
            
    for tup in zip(*[weeks[c] for c in keys + ['Week', 'CustomerInterest']]):
        tup, week, q = tup[:-2], tup[-2], tup[-1]
        tup = tup[0] if len(tup)==1 else tup
        seq_dict[tup][week] = q
    return seq_dict

def scale_features(df, scaler, num_cols):
    scaled = scaler.transform(df[num_cols])
    for i, col in enumerate(num_cols):
        df[col] = scaled[:,i]

def shift_right(seq, week, n_weeks):
    places = n_weeks - week - 1
    seq = np.roll(seq, places)
    seq[:places] = 0
    return seq

def roll_sequences(transactions, buysells, customers, isins, c, i, b, w, n_weeks):
    return [shift_right(transactions[(c,i,b)], w, n_weeks), 
            shift_right(buysells[(c,i)], w, n_weeks),
            shift_right(customers[c], w, n_weeks),
            shift_right(isins[i], w, n_weeks)]

from tqdm import tqdm_notebook

def extract_seqs(df, transactions, buysells, customers, isins, n_weeks):
    return np.array([roll_sequences(transactions, buysells, customers, 
                                    isins, c, i, b, w, n_weeks) \
                     for c,i,b,w in tqdm_notebook(zip(df.CustomerIdx, 
                     df.IsinIdx, df.BuySell, df.Week), total=len(df))])

def preprocess_catsconts(train, val, test, cat_cols, num_cols, scaler):
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
    scale_features(train, scaler, num_cols)
    scale_features(val, scaler, num_cols)
    scale_features(test, scaler, num_cols)

def preprocess(train, val, test, cat_cols, num_cols, seq_transactions, 
               seq_buysells, seq_customers, seq_isins, week_labels, scaler=None):
    
    n_weeks = len(week_labels)
    print('Extracting seqs...')
    train_seqs = extract_seqs(train, seq_transactions, seq_buysells, 
                              seq_customers, seq_isins, n_weeks)
    val_seqs = extract_seqs(val, seq_transactions, seq_buysells, 
                              seq_customers, seq_isins, n_weeks)
    test_seqs = extract_seqs(test, seq_transactions, seq_buysells, 
                              seq_customers, seq_isins, n_weeks)
        
    if scaler is None:
        scaler = StandardScaler().fit(pd.concat([train[num_cols], 
                                  val[num_cols], test[num_cols]]))

    preprocess_catsconts(train, val, test, cat_cols, num_cols, scaler)    
        
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
    
        
class StructuredNet(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, szs, drops,
                 rnn_hidden_sz, rnn_input_sz, rnn_n_layers, rnn_drop,
                 use_bn=True, out_sz=1):
        super().__init__()
        
        self.embs = nn.ModuleList([
            nn.Embedding(c, s) for c,s in emb_szs
        ])
        for emb in self.embs:
            self.emb_init(emb)
            
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont = n_emb, n_cont
        szs = [n_emb + n_cont + rnn_hidden_sz] + szs
        
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
        
        ## LSTM
        
        self.lstm = nn.LSTM(rnn_input_sz, rnn_hidden_sz, rnn_n_layers, 
                            dropout=rnn_drop)
        
        self.rnn_n_layers = rnn_n_layers
        self.rnn_hidden_sz = rnn_hidden_sz
    
    def forward(self, x_cat, x_cont, seqs, hidden):
        if self.n_emb != 0:
            x = [emb(x_cat[:,i]) for i,emb in enumerate(self.embs)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x2 = self.bn(x_cont)
            x = torch.cat([x, x2], 1) if self.n_emb != 0 else x2
            
        seqs = seqs.transpose(1,0).transpose(2,0) 
        output, hidden = self.lstm(seqs, hidden)
        x = torch.cat([x, output[-1]], 1) # last LSTM state
            
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
        
    def init_hidden(self, batch_sz, USE_CUDA):
        hidden = torch.zeros(self.rnn_n_layers, batch_sz, self.rnn_hidden_sz)
        cell = torch.zeros(self.rnn_n_layers, batch_sz, self.rnn_hidden_sz)
        if USE_CUDA:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)
        
def train_step(model, cats, conts, seqs, hidden, targets, 
               optimizer, criterion):
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
            hidden = model.init_hidden(len(cats), USE_CUDA)
            if USE_CUDA:
                cats, conts, seqs, target = cats.cuda(), conts.cuda(), \
                                    seqs.cuda(), target.cuda()
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
            hidden = model.init_hidden(len(cats), USE_CUDA)
            if USE_CUDA:
                cats, conts, seqs, target = cats.cuda(), conts.cuda(), \
                                    seqs.cuda(), target.cuda()
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