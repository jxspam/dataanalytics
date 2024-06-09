#My work throughout whole modules from Google Data Analytics Certificate

##Step 1: (Course 3) Exploratory Data Analysis (EDA)
###Process includes: discovering, structuring, cleaning, joining, 
###     validating, presenting

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv")

##1a. Discovering
print(df.head(10), # to display the first 10 rows of table
      df.info(), # to understand the type of variable of classes
                # and count the number of data within it
      df.describe(include = "all"), # to understand the general descriptive stats
                                     # in the dataframe
      df.shape # to know the number of rows and columns
      )

'''
#Stage 1a. Discovering the data

## Display first ten rows of the table
   Unnamed: 0  VendorID  ... improvement_surcharge total_amount
0    24870114         2  ...                   0.3        16.56
1    35634249         1  ...                   0.3        20.80
2   106203690         1  ...                   0.3         8.75
3    38942136         2  ...                   0.3        27.69
4    30841670         2  ...                   0.3        17.80
5    23345809         2  ...                   0.3        12.36
6    37660487         2  ...                   0.3        59.16
7    69059411         2  ...                   0.3        19.58
8     8433159         2  ...                   0.3         9.80
9    95294817         1  ...                   0.3        16.55

## Know the information for the data

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 22699 entries, 0 to 22698
Data columns (total 18 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----
 0   Unnamed: 0             22699 non-null  int64
 1   VendorID               22699 non-null  int64
 2   tpep_pickup_datetime   22699 non-null  object
 3   tpep_dropoff_datetime  22699 non-null  object
 4   passenger_count        22699 non-null  int64
 5   trip_distance          22699 non-null  float64
 6   RatecodeID             22699 non-null  int64
 7   store_and_fwd_flag     22699 non-null  object
 8   PULocationID           22699 non-null  int64
 9   DOLocationID           22699 non-null  int64
 10  payment_type           22699 non-null  int64
 11  fare_amount            22699 non-null  float64
 12  extra                  22699 non-null  float64
 13  mta_tax                22699 non-null  float64
 14  tip_amount             22699 non-null  float64
 15  tolls_amount           22699 non-null  float64
 16  improvement_surcharge  22699 non-null  float64
 17  total_amount           22699 non-null  float64
dtypes: float64(8), int64(7), object(3)
memory usage: 3.1+ MB
[11 rows x 18 columns] (22699, 18)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 22699 entries, 0 to 22698

 ### There don't have any null data row in the original csv datasheet

# Display the data for their discretion
[10 rows x 18 columns] None           Unnamed: 0      VendorID  ... improvement_surcharge  total_amount
count   2.269900e+04  22699.000000  ...          22699.000000  22699.000000
unique           NaN           NaN  ...                   NaN           NaN
top              NaN           NaN  ...                   NaN           NaN
freq             NaN           NaN  ...                   NaN           NaN
mean    5.675849e+07      1.556236  ...              0.299551     16.310502
std     3.274493e+07      0.496838  ...              0.015673     16.097295
min     1.212700e+04      1.000000  ...             -0.300000   -120.300000
25%     2.852056e+07      1.000000  ...              0.300000      8.750000
50%     5.673150e+07      2.000000  ...              0.300000     11.800000
75%     8.537452e+07      2.000000  ...              0.300000     17.800000
max     1.134863e+08      2.000000  ...              0.300000   1200.290000

'''
##1b. Structuring
df.sort_values(by=['fare_amount'], ascending=False) # sorting the dataframe as the priority

### Create a new time_duration column to calculate the time interval between pickup and dropff time
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df["time_duration"] = df['tpep_dropoff_datetime']- df['tpep_pickup_datetime']
df["time_duration_hours"] = df["time_duration"].dt.total_seconds() / 3600

### Change to the rate code ID into the categorical variable
import datetime as dt
df['RatecodeID'] = df['RatecodeID'].astype('category')
df['weekday'] = df['tpep_pickup_datetime'].dt.day_name().str[:3] #Note: Day_name: Monday, Tuesday... Sunday
df['hour'] = df['tpep_pickup_datetime'].dt.hour # Categorise into the hour

### Only taking those data needed for the data insight
df = df.loc[:,['hour','time_duration','time_duration_hours','weekday','trip_distance',\
               "RatecodeID",'payment_type','fare_amount','VendorID','passenger_count']]
            
            
##1c. Cleaning (to ensure the quality data is remained)
###i: remove the null data and duplicated data
df = df.drop_duplicates()
df = df.ffill()
# df.any(axis=1) to determine if there has all null values for a row

###ii: filter the extraordinary data
df = df[df["fare_amount"]>0] 
df = df[df["trip_distance"]>0]
df = df[df["RatecodeID"].isin([1,2,3,4,5,6])]

#1d. Joining - skip

#1e. Validating

print(df.info())
'''
## Make sure that the data type is updated, 
   and there is another row inserted in the table

Data columns (total 20 columns):
 #   Column                 Non-Null Count  Dtype
---  ------                 --------------  -----
 0   Unnamed: 0             22699 non-null  int64
 1   VendorID               22699 non-null  int64
 2   tpep_pickup_datetime   22699 non-null  datetime64[ns]
 3   tpep_dropoff_datetime  22699 non-null  datetime64[ns]
 4   passenger_count        22699 non-null  int64
 5   trip_distance          22699 non-null  float64
 6   RatecodeID             22699 non-null  int64
 7   store_and_fwd_flag     22699 non-null  object
 8   PULocationID           22699 non-null  int64
 9   DOLocationID           22699 non-null  int64
 10  payment_type           22699 non-null  int64
 11  fare_amount            22699 non-null  float64
 12  extra                  22699 non-null  float64
 13  mta_tax                22699 non-null  float64
 14  tip_amount             22699 non-null  float64
 15  tolls_amount           22699 non-null  float64
 16  improvement_surcharge  22699 non-null  float64
 17  total_amount           22699 non-null  float64
 18  time_duration          22699 non-null  timedelta64[ns]
 19  time_duration_hours    22699 non-null  float64
dtypes: datetime64[ns](2), float64(9), int64(7), object(1), timedelta64[ns](1)
memory usage: 3.5+ MB
'''

print(df.isna().sum())
'''
## To check there's no null number
None
hour                   0
time_duration          0
time_duration_hours    0
weekday                0
trip_distance          0
VendorID               0
passenger_count        0
dtype: int64
'''

##1f. Presenting  for initial insight about the questions    
#### Question 1: The total fare amount across the days of week
weekday_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
sns.boxplot(data = df, 
            x = 'weekday',
            y = 'fare_amount', 
            order = weekday_order, # Set the order for the graph
            showfliers = False 
            )

plt.xlabel("Weekday")
plt.ylabel("Total Fare Amount")
plt.title("Fare amount for taxi cab per weekday in 2017-2020")
plt.show()

# ###Question 2: The amount of trips in each hour for day and each day
df_count_by_weekday = df.groupby(['weekday']).size().reset_index(name = 'count')
df_count_by_hour = df.groupby(['weekday', 'hour']).size().reset_index(name = 'count')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4)) #figsize in
sns.barplot(data=df_count_by_hour,
            x='weekday',
            y='count',
            hue='weekday',ax=axes[0])
axes[0].set_xlabel("Weekday", size = 12)  
axes[0].set_ylabel("Count", size = 12)
axes[0].set_title("The Amount of Trips in each Weekday")
            
sns.barplot(data=df_count_by_hour,
            x='hour',
            y='count',
            hue='weekday',ax=axes[1])
axes[1].set_xlabel("Hour per each Weekday")
axes[1].set_ylabel("Count")
axes[1].set_title("The Amount of Trips in each Hour for each Weekday")

plt.title("The amount of trips in each hour for day and each day")
plt.tight_layout()
plt.show()

###Question 3: The average total fare amount in each hour for day
df_fare_by_hour = df.groupby(['weekday', 'hour'])['fare_amount']\
        .agg('mean').reset_index(name='mean_fare_amount')
print(df_fare_by_hour.head())
sns.barplot(data=df_fare_by_hour,
            x="hour",
            y="mean_fare_amount",
            hue="weekday")
plt.xlabel("Weekday")
plt.ylabel("Average Fare Amount")
plt.title("Average Fare Amount in each Hour for each Weekday")
plt.show()

            
###Question 4: Fare amount against the duration of driving time
df = df[(df['fare_amount'] <= 250) & (df['time_duration_hours'] > 0) & (df['time_duration_hours'] <= 3)]
sns.regplot(data=df, 
            x="time_duration_hours",
            y='fare_amount')
plt.xlabel("Time Duration of Trip (h)")
plt.ylabel("Fare Amount")
plt.title("Fare Amount against Time Duration of Trip")
plt.show()

# The data is filtered out the trip with fare amount of 52
df = df[(df["fare_amount"])!=52]

sns.regplot(data=df, 
            x="time_duration_hours",
            y='fare_amount')
plt.xlabel("Time Duration of Trip (h)")
plt.ylabel("Fare Amount")
plt.title("Fare Amount against Time Duration of Trip")
plt.show()

###Question 5: Fare amount against the trip distance
sns.regplot(data = df,
            x='trip_distance',
            y='fare_amount') 
plt.xlabel("Trip distance (miles)")
plt.ylabel("Fare Amount")
plt.title("Fare Amount against Trip Distance")
plt.show()


###Question 6: Average fare amount for RateCodeID
df_fare_by_RatecodeID = df[["RatecodeID","fare_amount"]].\
                        groupby(["RatecodeID"])["fare_amount"].agg('mean').reset_index(name='mean_fare_amount')
sns.barplot(data=df_fare_by_RatecodeID,
            x="RatecodeID",
            y="mean_fare_amount")
plt.xlabel("RatecodeID")
plt.ylabel("Average Fare Amount")
plt.title("Average Fare Amount in each Rate Code ID")
plt.show()


#-------------------------------------------------------------------------------------------------

#Step 2: (Course 4) Hypothesis Testing 

from scipy import stats
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import ols

###Question 1: Does the vendor ID affect the fare amount? - Using one way ANOVA test 
###for the 4 categorical groups

# Construct simple linear regression model, and fit the model
model = ols(formula = "fare_amount ~ C(RatecodeID)", data = df).fit()

# Check for the ANOVA assumptions
##1. Identically independent distributed data
##2. The variances of the groups are the same
##3. The residuals are distributed normally

residuals = model.resid

fig, axes = plt.subplots(1, 2, figsize = (8,4))

sns.histplot(residuals, ax = axes[0])
axes[0].set_xlabel("Residual Value")
axes[0].set_title("Histogram of Residuals")

sm.qqplot(residuals, line = "s", ax = axes[1])
axes[1].set_title("Normal Q-Q Plot")

plt.tight_layout()
plt.show()


# Perform the Kruskal-Wallis test
RatecodeID = df['RatecodeID']
groups = [df['fare_amount'][RatecodeID == group] for group in RatecodeID.unique()]
kruskal_result = stats.kruskal(*groups)

print("Kruskal-Wallis H-test result:", kruskal_result)

'''
# Kruskal-Wallis H-test result: 
KruskalResult(statistic=208.62043481421924, pvalue=5.785403289637155e-45)
'''


#-------------------------------------------------------------------------------------------------
##Step 3: (Course 5) Linear  Regression
# Modify the data for the df, for the linear regression purpose
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df_linear_reg = df.copy()

# Remain the "fare_amount", "trip_distance", "time_duration_hours" for the linear regression analysis
df_linear_reg = df_linear_reg[["fare_amount", "trip_distance", "time_duration_hours"]]
sns.pairplot(df_linear_reg)
plt.show()

# Log transformation for skewed data
df['log_fare_amount'] = np.log1p(df['fare_amount'])
df['log_trip_distance'] = np.log1p(df['trip_distance'])
df['log_time_duration_hours'] = np.log1p(df['time_duration_hours'])

# Prepare the data for regression
X = df[['log_trip_distance', 'log_time_duration_hours']]
X = sm.add_constant(X)
y = df['log_fare_amount']

# Fit the model
model = sm.OLS(y, X).fit()
print(model.summary())

# Check for multicollinearity
vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

'''
                            OLS Regression Results
==============================================================================
Dep. Variable:        log_fare_amount   R-squared:                       0.964
Model:                            OLS   Adj. R-squared:                  0.964
Method:                 Least Squares   F-statistic:                 2.907e+05
Date:                Tue, 28 May 2024   Prob (F-statistic):               0.00
Time:                        11:01:52   Log-Likelihood:                 19026.
No. Observations:               21992   AIC:                        -3.805e+04
Df Residuals:                   21989   BIC:                        -3.802e+04
Df Model:                           2
Covariance Type:            nonrobust
===========================================================================================       
                              coef    std err          t      P>|t|      [0.025      0.975]       
-------------------------------------------------------------------------------------------       
const                       1.4382      0.001    963.815      0.000       1.435       1.441       
log_trip_distance           0.5123      0.002    246.584      0.000       0.508       0.516       
log_time_duration_hours     2.0798      0.009    222.264      0.000       2.061       2.098       
==============================================================================
Omnibus:                    12249.193   Durbin-Watson:                   1.976
Prob(Omnibus):                  0.000   Jarque-Bera (JB):         13158905.166
Skew:                           1.190   Prob(JB):                         0.00
Kurtosis:                     122.811   Cond. No.                         21.5
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.       
                   Feature       VIF
0                    const  4.717770
1        log_trip_distance  2.920164
2  log_time_duration_hours  2.920164
''' 

