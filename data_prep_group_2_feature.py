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
# print "reading csv"

# df = pd.read_csv('data/Train.csv',usecols=['SalesID','MachineID','ModelID','YearMade','MachineHoursCurrentMeter','SalePrice', 'Enclosure', 'Hydraulics'])

df = pd.read_csv('data/Train.csv',usecols=['ProductSize','saledate','YearMade','MachineHoursCurrentMeter','SalePrice'])

# plt.hist(df['YearMade'],bins=20)
# plt.show()

def data_prep_group_1(df):

    # print "dropping year"

    df = df.drop(df[df['YearMade'] < 1800].index,axis=0)

    # print "calibrate min year"
    # min_year = 1919
    df['SaleYear'] = df['saledate'].map(lambda x: x[-9:-5]).astype(int)
    df['Age'] =  (df['SaleYear'] - df['YearMade']).astype(int)

    # print "fill Meter NaN to 0"
    df['MachineHoursCurrentMeter'].fillna(value=0.0,inplace=True)



    # print "dummifying"
    # df = dummify(df,['ModelID'])

    return df

new_df = data_prep_group_1(df)

def dummify(df, columns):
    return pd.get_dummies(df, columns=columns)


def linear_model_sklearn():
    new_df = data_prep_group_1(df)
    sample = new_df.sample(frac=1)
    # print "splitting X,y"
    y = sample['SalePrice'].values
    X = sample[['YearMade','MachineHoursCurrentMeter','Age']].values

    # print X

    # print "getting sample"

    # print "fitting"
    model = LinearRegression(n_jobs=-1).fit(X,y)

    # print "scoring"
    # print model.score(X,y)
    return model

def linear_model_sm():
    new_df = data_prep_group_1(df)
    sample = new_df.sample(frac=1)
    # print "splitting X,y"
    y = sample['SalePrice']
    X = sample[['YearMade','MachineHoursCurrentMeter','Age']]

    X = sm.add_constant(X)
    model = sm.OLS(y,X)
    model = model.fit()
    # print "summary"
    print(model.summary())
    return model

# mod = linear_model_sklearn()
mod = linear_model_sm()
# df_test = pd.read_csv('data/Test.csv',usecols=['SalesID','saledate','YearMade','MachineHoursCurrentMeter'])
# df_test = data_prep_group_1(df_test)
# sales_id = df_test['SalesID'].values
# X_test = df_test[['YearMade','MachineHoursCurrentMeter','Age']].values
# pred = mod.predict(X_test)
#
# print 'SalesID,SalePrice'
# for i in xrange(X_test.shape[0]):
#     print "{},{}".format(sales_id[i], pred[i])
print mod.params
