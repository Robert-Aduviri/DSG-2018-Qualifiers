from datetime import date
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
    
def date_diff(d1, d2):
    '''#Days between d1 and d2, expressed as integers'''
    return (date(d1 // 10000, (d1 // 100) % 100, d1 % 100) - \
            date(d2 // 10000, (d2 // 100) % 100, d2 % 100)).days
    
def days_since(day_df, trades, keys, nan_date=20170701):
    '''Get number of days between last *keys* and day_df date'''
    last_trades = pd.Series(trades.drop_duplicates(keys, keep='first') \
            .set_index(keys)['TradeDateKey']).to_dict()
    return day_df.apply(lambda r: date_diff(r['TradeDateKey'],
            last_trades.get(tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]],
            nan_date)), axis=1)
    
# Count without considering weekdays
def add_datediffs(day_df, trades):
    """Adds datediffs features to a dataset (representing a single day)
    from the information of trades. Adds #DaysSinceBuySell (the corresponding
    one) #DaysSinceTransaction (either buy or sell), #DaysSinceCustomerActivity
    (since last customer interaction) #DaysSinceBondActivity (since last bond
    interaction)"""
    trades = trades[trades.CustomerInterest == 1]
    date = day_df['TradeDateKey'].unique()[0]
    trades = trades[trades.TradeDateKey < date]
    trades = trades.sort_values('TradeDateKey', ascending=False)
    
    day_df['DaysSinceBuySell'] = days_since(day_df, trades, 
                                            ['CustomerIdx', 'IsinIdx', 'BuySell'])
    day_df['DaysSinceTransaction'] = days_since(day_df, trades, 
                                            ['CustomerIdx', 'IsinIdx'])
    day_df['DaysSinceCustomerActivity'] = days_since(day_df, trades, ['CustomerIdx'])
    day_df['DaysSinceBondActivity'] = days_since(day_df, trades, ['IsinIdx'])
    
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
    
    
    