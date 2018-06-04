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
    date = day_df['TradeDateKey'].unique()[0]
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
    