from datetime import date, timedelta
from bisect import bisect_right
import numpy as np
import pandas as pd

def apply_cats(df, trn):
    """Changes any columns of strings in df (DataFrame) into categorical variables
    using trn (DataFrame) as a template for the category codes (inplace)."""
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)
            
def make_val_set(day_df, challenge):
    """Makes a validation set from the information of the given day (DataFrame)
    filling complementary BuySell labels and including the data points from the
    original test set in challenge (DataFrame)"""
    val = {}
    date = sorted(day_df['TradeDateKey'].unique())[0]
    for idx, row in challenge.iterrows():
        val[(date, row['CustomerIdx'], 
             row['IsinIdx'], row['BuySell'])] = 0
    # 0s first, then 1s
    for idx, row in day_df.sort_values('CustomerInterest').iterrows():
        key = (date, row['CustomerIdx'], 
               row['IsinIdx'], row['BuySell'])
        val[key] = row['CustomerInterest']
        key = list(key)
        key[-1] = 'Sell' if row['BuySell'] == 'Buy' else 'Buy'
        key = tuple(key)
        if key not in val:
            val[key] = 0
    val = pd.DataFrame(pd.Series(val)).reset_index()
    val.columns = ['TradeDateKey', 'CustomerIdx', 'IsinIdx', 'BuySell',
                   'CustomerInterest']
    return val

def get_weeks(day_from=20150105, num_weeks=200):
    return [date2num(num2date(day_from) + timedelta(i * 7)) for i in range(num_weeks)]

def week_num(weeks, day):
    return bisect_right(weeks, day) - 1

def week_day(weeks, num):
    return weeks[num]

def num2date(n):
    return date(n // 10000, (n // 100) % 100, n % 100)
    
def date2num(d):
    return int(str(d).replace('-', ''))    
    
def date_diff(d1, d2):
    '''#Days between d1 and d2, expressed as integers'''
    return (num2date(d1) - num2date(d2)).days
    
def days_since(day_df, trades, keys, nan_date):
    '''Get number of days between last *keys* and day_df date'''
    last_trades = pd.Series(trades.drop_duplicates(keys, keep='first') \
            .set_index(keys)['TradeDateKey']).to_dict()
    return day_df.apply(lambda r: date_diff(r['TradeDateKey'],
            last_trades.get(tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]],
            nan_date)), axis=1)
    
# Count without considering weekdays
def add_datediffs(day_df, trades, nan_date=20170701):
    """Adds datediffs features to a dataset (representing a single day/week)
    from the information of trades. Adds #DaysSinceBuySell (the corresponding
    one) #DaysSinceTransaction (either buy or sell), #DaysSinceCustomerActivity
    (since last customer interaction) #DaysSinceBondActivity (since last bond
    interaction)"""
    trades = trades[trades.CustomerInterest == 1]
    date = sorted(day_df['TradeDateKey'].unique())[0]
    trades = trades[trades.TradeDateKey < date]
    trades = trades.sort_values('TradeDateKey', ascending=False)
    
    day_df['DaysSinceBuySell'] = days_since(day_df, trades, 
                                    ['CustomerIdx', 'IsinIdx', 'BuySell'], nan_date)
    day_df['DaysSinceTransaction'] = days_since(day_df, trades, 
                                    ['CustomerIdx', 'IsinIdx'], nan_date)
    day_df['DaysSinceCustomerActivity'] = days_since(day_df, trades, ['CustomerIdx'], 
                                    nan_date)
    day_df['DaysSinceBondActivity'] = days_since(day_df, trades, ['IsinIdx'],
                                    nan_date)

def days_count(day_df, trades, keys):
    '''Get frequency *keys* in historical trades before day_df'''
    day_counter = trades.groupby(keys).size().to_dict()
    return day_df.apply(lambda r: \
            day_counter.get(tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]], 
            0), axis=1)
    
def add_dayscount(day_df, trades):
    '''Adds dayscount features to a dataset (representing a single day/week)
    from the information of trades'''
    trades = trades[trades.CustomerInterest == 1]
    date = sorted(day_df['TradeDateKey'].unique())[0]
    trades = trades[trades.TradeDateKey < date]
    
    day_df['DaysCountBuySell'] = days_count(day_df, trades,
                                    ['CustomerIdx', 'IsinIdx', 'BuySell'])
    day_df['DaysCountTransaction'] = days_count(day_df, trades,
                                    ['CustomerIdx', 'IsinIdx'])
    day_df['DaysCountCustomerActivity'] = days_count(day_df, trades, ['CustomerIdx'])
    day_df['DaysCountBondActivity'] = days_count(day_df, trades, ['IsinIdx'])
    
def composite_rating_cmp(x, y):
    if x[0] != y[0]: # A vs B
        return -1 if x[0] < y[0] else 1
    len_x = len([c for c in x if c.isalpha()])
    len_y = len([c for c in y if c.isalpha()])
    if len_x != len_y: # AAA vs AA
        return -1 if len_x > len_y else 1
    if x != y: # BB+ BB-
        if '+' in x:
            return -1 
        elif '+' in y:
            return 1
        else:
            return -1 if len(x) < len(y) else 1
    return 0
    
def preprocessing_pipeline(df, customer, isin, trade):
    df = pd.merge(df, customer, how='left', on='CustomerIdx')
    df = pd.merge(df, isin, how='left', on='IsinIdx')
    #id_cols = ['CustomerIdx', 'IsinIdx']
    #target_col = 'CustomerInterest'
    #num_cols = ['ActualMaturityDateKey', 'IssueDateKey', 'IssuedAmount', 'TradeDateKey']
    #cat_cols = ['BuySell', 'Sector', 'Subsector', 'Region_x', 'Country', 'TickerIdx',
    #            'Seniority', 'Currency', 'ActivityGroup', 'Region_y', 'Activity',
    #            'RiskCaptain', 'Owner', 'CompositeRating', 'IndustrySector',
    #            'IndustrySubgroup', 'MarketIssue', 'CouponType']
    return df


##### SMA ##### by Kervy

def calculate_SMA(df, period, start=0, column='Price'):
    """
        Returning the First SMA to calculate the first EMA
    """
    return df.loc[start:period + start - 1][column].sum() / period

def calculate_EMA(prev_EMA, price, multiplier):
    """
        Returning the EMA for t time
    """
    return (price - prev_EMA) * multiplier + prev_EMA

def fill_EMA(df, period=20, name_column='EMA_Price_Short_term', column='Price'):
    """
        Exponential moving averages (EMAs) reduce the lag by applying more weight to recent prices
    """
    first_SMA = calculate_SMA(df, period, column=column)
    multiplier= (2.0 / (period + 1))    
    df[name_column] = np.nan
    for ix, _ in df.iterrows():
        if ix < period - 1:
            continue
        elif ix == period - 1:
            df.set_value(ix, name_column, first_SMA)
            prev_EMA = first_SMA
        else:
            if np.isnan(df.loc[ix][column]):
                df.set_value(ix, column, (df.loc[ix-1][column] + df.loc[ix+1][column]) / 2)
            actual_EMA = calculate_EMA(prev_EMA, df.loc[ix][column], multiplier)
            prev_EMA = actual_EMA
            df.set_value(ix, name_column, actual_EMA)
##### MODEL ######

import time, pprint    
from sklearn.metrics import roc_auc_score
pp = pprint.PrettyPrinter(indent=3)

# globals: [cat_indices]
def fit_model(model, model_name, X_trn, y_trn, X_val, y_val, early_stopping_rounds, cat_indices):
    if X_val is not None:
        if model_name in ['XGBClassifier', 'LGBMClassifier']:
            model.fit(X_trn, y_trn, 
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=early_stopping_rounds,
                      eval_metric='auc')
        elif model_name == 'CatBoostClassifier':
            model.fit(X_trn, y_trn, 
                      eval_set=[(X_val, y_val)],
                      use_best_model=True,
                      cat_features=cat_indices)
        else:
            model.fit(X_trn, y_trn)
    else:
        if model_name == 'CatBoostClassifier':
            model.fit(X_trn, y_trn, 
                      cat_features=cat_indices)
        else:
            model.fit(X_trn, y_trn)
        
def calculate_metrics(model, metrics, X_trn, y_trn, X_val, y_val):
    metric_function = {'auc': roc_auc_score}
    dset = {'trn': {'X': X_trn, 'y': y_trn},
            'val': {'X': X_val, 'y': y_val}}
    
    for d in dset:
        if dset[d]['X'] is not None:
            y_pred = model.predict_proba(dset[d]['X'])[:,1]
            for m in metrics:
                metrics[m][d] += [metric_function[m](dset[d]['y'], y_pred)]
        else:
            for m in metrics:
                metrics[m][d] += [0] # no val set
                
    pp.pprint(metrics)
    print()
    
def run_model(model, X_train, y_train, X_val, y_val, X_test, 
              metric_names, results=None, dataset_desc='', params_desc='',
              early_stopping_rounds=None, cat_indices=None):
    model_name = str(model.__class__).split('.')[-1].replace('>','').replace("'",'')
    print(model_name, '\n')
    if results is None: results = pd.DataFrame()
    metrics = {metric: {'trn': [], 'val': []} for metric in metric_names}
    
    start = time.time()
    
    fit_model(model, model_name, X_train, y_train, X_val, y_val, early_stopping_rounds, cat_indices)
    calculate_metrics(model, metrics, X_train, y_train, X_val, y_val)
    
    y_test = model.predict_proba(X_test)[:,1] if X_test is not None else None    
            
    end = time.time()
    means = {f'{d}_{m}_mean': np.mean(metrics[m][d]) for m in metrics \
                                                     for d in metrics[m]}
    metadata = {'model': model_name, 'dataset': dataset_desc,
                'params': params_desc, 'time': round(end - start, 2)}
    pp.pprint(means)
    results = results.append(pd.Series({**metadata, **means}),
                             ignore_index=True)
    return y_test, metrics, results, model