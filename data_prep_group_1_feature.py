import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import numpy as np
import statsmodels.api as sm

  # unique identifier of a particular sale of a machine at auction

  # identifier for a particular machine;  machines may have multiple sales

  # identifier for a unique machine model (i.e. fiModelDesc)

  # year of manufacturer of the Machine

  # current usage of the machine in hours at time of sale (saledate);  null or 0 means no hours have been reported for that sale
print "reading csv"

# df = pd.read_csv('data/Train.csv',usecols=['SalesID','MachineID','ModelID','YearMade','MachineHoursCurrentMeter','SalePrice', 'Enclosure', 'Hydraulics'])

df = pd.read_csv('data/Train.csv',usecols=['SalesID','ProductSize','MachineID','ModelID','YearMade','MachineHoursCurrentMeter','SalePrice'])

# plt.hist(df['YearMade'],bins=20)
# plt.show()

def data_prep_group_1_without_adjusting_hours(df):

    print "dropping year"

    df = df.drop(df[df['YearMade'] < 1800].index,axis=0)

    print "calibrate min year"
    min_year = 1919
    df['YearMade'] = df['YearMade'] - min_year

    print "fill Meter NaN to 0"
    df['MachineHoursCurrentMeter'].fillna(value=0.0,inplace=True)

    print "dummifying"
    df = dummify(df,['ModelID'])

    return df

def data_prep_group_1_with_adjusting_hours(df):

    print "dropping year"

    df = df.drop(df[df['YearMade'] < 1800].index,axis=0)

    print "calibrate min year"
    min_year = 1919
    df['YearMade'] = df['YearMade'] - min_year

    print "fill Meter NaN to 0"
    df['MachineHoursCurrentMeter'].fillna(value=0.0,inplace=True)

    # df['New'] = (df['MachineHoursCurrentMeter']==0) & (df['UsageBand']=='High')
    # df['New'] = df['New'].astype(int)

    print "dummifying"
    df = dummify(df,['ModelID'])

    return df

def dummify(df, columns):
    return pd.get_dummies(df, columns=columns)



def linear_model_sklearn_new():
    new_df = data_prep_group_1_with_adjusting_hours(df)
    sample = new_df.sample(frac=.01)
    print "splitting X,y"
    y = sample['SalePrice'].values
    X = sample.drop(['SalePrice','UsageBand'],axis=1).values

    print "getting sample"

    print "fitting"
    model = LinearRegression(n_jobs=-1).fit(X,y)

    print "scoring"
    print model.score(X,y)

def linear_model_sklearn():
    new_df = data_prep_group_1_with_adjusting_hours(df)
    sample = new_df.sample(frac=.5)
    print "splitting X,y"
    y = sample['SalePrice'].values
    X = sample.drop(['SalePrice'],axis=1).values

    print "getting sample"

    print "fitting"
    model = LinearRegression(n_jobs=-1).fit(X,y)

    print "scoring"
    print model.score(X,y)

def linear_model_sm():
    # print "fitting"
    X = sm.add_constant(X)
    model = sm.OLS(y,X)
    model = model.fit()
    print "summary"
    print(model.summary())

mod = linear_model_sklearn()
