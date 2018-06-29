## TODO

#### Data preprocessing
- [x] join customer and isin to trade
- [ ] add temporal data (RNN bond / macro prices until current point (fixed time window / variable scales (daily, weekly, monthly)))
- [ ] add historical data (RNN previous bonds)
- [ ] convert currency
- [x] normalize data
- [x] fill nan

#### Feature Engineering
- [ ] add date features 
- [ ] group minority classes
- [x] add datediffs features
- [x] add dayscount features
- [x] mean encoding (only catboost version)
- [x] CompositeRating (ordinal label)
- [x] Actual MaturityDateKey - IssueDateKey (in days)
- [x] Actual MaturityDateKey - Current Day (in days)
- [x] Relative variation of bond price / yield (last n weeks)
- [x] Relative variation of bond currency exchange value (last n weeks)
- [ ] # bonds bought/sold in the previous week

#### Cross validation
- [x] setup cross validation considering future decisions for target optimization
- [x] add previous weeks for training (8 weeks)
- [x] remove 20170701 nan_date
- [ ] consider / omit weekdays
- [x] test more weeks and compare predictions

#### EDA
- [x] histogram of #trades per customer / bond
- [ ] price when customerInterest = 1

#### Models
- [x] LightGBM
- [x] Catboost
- [x] Deep Neural Net
- [ ] RNN for market and macro (metadata as init-hidden via dense layer, day/month/macro/market as sequence input, interest as sequence output)
- [x] Matrix Factorization (item = Bond+BuySell)
- [x] Cat embeddings for categorical data
- [x] Add cats+conts at each timestep / at the beginning / at the end
- [ ] Train with / without zero sequences

- [ ] Train with targets of previous timesteps
- [ ] Fast.ai functions
- [ ] Control overfitting
- [ ] More dense layers / dropout / ReLU activations at the end
- [ ] Get macro / market data
- [ ] Recommender system based on embeddings

- [ ] Add Customer and Bond embedding
- [ ] Use sequence as concatenated feature

```
torch.utils.data.ConcatDataset(datasets)

users,movies = cats[:,0],cats[:,1]
u2,m2 = self.u(users) , self.m(movies)
x = self.drop1(torch.cat([u2,m2], 1)) # drop initialized weights
x = self.drop2(F.relu(self.lin1(x))) # drop 1st linear + nonlinear wt
r = F.sigmoid(self.lin2(x)) * (max_rating - min_rating) + min_rating   
```


