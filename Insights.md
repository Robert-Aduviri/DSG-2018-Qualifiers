## Insights List
- Activity (# trades) increases at the end of each month, there is an atypical peak (4x) in the first month
- The increase of activity is because of the holding reports (legal obligation) (just consider interest=1 events)
- There is no activity on weekends and holidays
- There should be around 4.2K positive labels in test set
- There is historical information for all customers in test
- There is historical information for all bonds in test
- There is historical information for all (customer,bond) pairs in test
- There is metadata for all customers in test
- There is metadata for all bonds in test

| Dataset | Rows | Columns | 
| ------- | ---- | ------- |
| challenge  |   484,758 |   6 |
| customer   |     3,471 |   5 |
| isin       |    27,411 |  17 |
| market     | 9,867,747 |   5 |
| macro      |       877 | 112 |
| trade      | 6,762,021 |   8 |
| submission |   484,758 |   2 |

### Nulls

| Dataset  | Column | % Nulls | Nulls |
| -------  | ------ | ------- | ----- | 
| trade    | Price  | 68.29%  | 4,617,933 |
| customer | Subsector | 10.14% | 352 | 
| isin     | IndustrySector | 0.02% | 5 |
| isin     | IndustrySubgroup | 0.02% | 5 |
| isin     | MarketIssue | 0.06% | 17 |
| macro | 30 columns | 0.11% - 3.31% | 1 - 29 |

### Cumulative frequencies

#### #Trades / Customers

![Image](https://raw.githubusercontent.com/Robert-Alonso/DSG-2018/master/assets/trades-customer.png?token=AFjmgS0xTYKedt2pmJhJdv8d7j3a2PI-ks5bHAe0wA%3D%3D)

#### #Trades / Bonds

![Image](https://raw.githubusercontent.com/Robert-Alonso/DSG-2018/master/assets/trades-bond.png?token=AFjmgWM0aBWq_o2EIzE60PIjZgd3S8-1ks5bHAf9wA%3D%3D)

#### #Trades / (Customers,Bonds)

![Image](https://raw.githubusercontent.com/Robert-Alonso/DSG-2018/master/assets/trades-customer%2Bbond.png?token=AFjmgeN5Epsubr_Jp2bponcxeNr34L9Qks5bHAgHwA%3D%3D)
