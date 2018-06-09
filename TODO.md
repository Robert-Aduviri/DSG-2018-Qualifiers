## TODO

#### Data preprocessing
- [x] join customer and isin to trade
- [ ] add temporal data (RNN bond / macro prices until current point (fixed time window / variable scales (daily, weekly, monthly)))
- [ ] add historical data (RNN previous bonds)
- [ ] convert currency

#### Feature Engineering
- [ ] add date features 
- [ ] group minority classes
- [x] add datediffs features
- [x] add dayscount features
- [x] mean encoding (only catboost version)
- [ ] CompositeRating (ordinal label)
- [ ] Actual MaturityDateKey - IssueDateKey (in days)
- [ ] Actual MaturityDateKey - Today (in days)
- [ ] 

#### Cross validation
- [x] setup cross validation considering future decisions for target optimization
- [x] add previous weeks for training (8 weeks)
- [ ] remove 20170701 nan_date
- [ ] consider / omit weekdays
- [ ] test more weeks and compare predictions

#### EDA
- [x] histogram of #trades per customer / bond
- [ ] price when customerInterest = 1
