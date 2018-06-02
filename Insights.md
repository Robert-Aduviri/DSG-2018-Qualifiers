## Insights List
- Activity (# trades) increases at the end of each month, there is an atypical peak (4x) in the first month
- There is historical information for all customers in test
- There is historical information for all bonds in test
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
| macro | 30 cols | 0.11% - 3.31% | 1 - 29 |


