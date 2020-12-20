# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import math
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder


def plotCorrMatrix(df):
    with pd.option_context('display.max_columns', None):
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df.corr(), annot=True, lineWIdths=0.5, ax = ax)
        
def plotOutliers(train, xcol, ycol):
    fig, ax = plt.subplots()
    ax.scatter(train[xcol], train[ycol])
    plt.ylabel(ycol, fontsize=13)
    plt.xlabel(xcol, fontsize=13)
    plt.show()


"""
check if target is normally distributed
"""
def plotSalePrice(train):
    
    """  
    If one tail is longer than another, the distribution is skewed.
    These distributions are sometimes called asymmetric or asymmetrical distributions
    """
    
    sns.distplot(train['SalePrice'] , fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(train['SalePrice'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    
    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    
    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(train['SalePrice'], plot=plt)
    plt.show()

"""
since target variable is skewed, we need to make it more normally distributed
"""
def targetLogTrans(target, plot = False):
    #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    target = np.log1p(target)
    
    if plot:
        #Check the new distribution 
        sns.distplot(target , fit=norm);
    
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(target)
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    

    

    
    if plot:
        #Now plot the distribution
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                    loc='best')
        plt.ylabel('Frequency')
        plt.title('target distribution')
        
        #Get also the QQ-plot
        fig = plt.figure()
        res = stats.probplot(target, plot=plt)
        plt.show()
    
    return target
    
    
    


def saveResult(model, dfTestPredict , ids, output='results.csv'):
    """
    CHOOSE MODEL AND SAVE RESULTS
    """
    
       
    res = model.predict(dfTestPredict)

    res = np.expm1(res)
    
    pdResult = pd.DataFrame(list(zip(ids, res)), columns=['Id', 'SalePrice'])
    
    #, float_format='%.15f'
    pdResult.to_csv(output, index=False)
    print("saved {} rows".format(pdResult.shape[0]))
    
    
# Function responsible for checking our model's performance on the test data
def testSetResultsClassifier(classifier, x_test, y_test, model_name =''):
    predictions = classifier.predict(x_test)
    
    results = []

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    results.append(r2)
    results.append(mse)
    results.append(rmse)
    results.append(mae)
    
    print("\n\n#---------------- Test set results ({}) ----------------#\n".format(model_name))
    print("r2_score, mse, rmse, mean absolute error:")
    print(results)
    
    return results


def processData(train, test, plotRes = False):
    
    #concatenate train and test in the same dataframe
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    missing_data.head(20)
    
    if plotRes:
        f, ax = plt.subplots(figsize=(15, 12))
        plt.xticks(rotation='45')
        sns.barplot(x=all_data_na.index, y=all_data_na)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        plt.show()
        
        
    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    
    all_data["Alley"] = all_data["Alley"].fillna("None")
    
    all_data["Fence"] = all_data["Fence"].fillna("None")
    
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
    
    #Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
        
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
        
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')
        
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    
    # df.mode() : The value that occurs most frequently
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    
    all_data = all_data.drop(['Utilities'], axis=1)
    
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    
    
    #MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    
    
    #Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    
    
    #Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(all_data[c].values)) 
        all_data[c] = lbl.transform(list(all_data[c].values))
        
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    
    
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(10)
    
    skewness = skewness[abs(skewness) > 0.75]
    
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        #all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)
    
    
    all_data = pd.get_dummies(all_data)
    
    
    all_data['TotalBsmtSF'] = pd.qcut(all_data['TotalBsmtSF'], q=4).cat.codes

    all_data['1stFlrSF'] = pd.qcut(all_data['1stFlrSF'], q=4).cat.codes
    
    all_data['GrLivAreaCol'] = pd.qcut(all_data['GrLivArea'], q=8).cat.codes
    
    #all_data['YearBuiltCol'] = pd.qcut(all_data['YearBuilt'], q=5).cat.codes
    
    #all_data['OpenPorchSFCol'] = pd.qcut(all_data['OpenPorchSF'], q=2).cat.codes
    
    #all_data['WoodDeckSFCol'] = all_data['WoodDeckSF'] > 0
    
    #all_data['WoodDeckSFCol'] = (all_data['WoodDeckSF'] < 1) & (all_data['OpenPorchSF'] < 1 ) 
    
    
    return all_data[:ntrain], all_data[ntrain:]
    
    


np.set_printoptions(suppress=True)
"""
LOAD DATASET
"""
pathTrain = os.path.join('data', 'train.csv')
pathTest = os.path.join('data', 'test.csv')
df = pd.read_csv(pathTrain)
dfTest = pd.read_csv(pathTest)



df.drop(['Id'], axis=1)
dfTest.drop(['Id'], axis=1)

ids = dfTest['Id']





#Deleting outliers
df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)

df = df.drop(df[(df['1stFlrSF']>3000)].index)



#plotOutliers(df, 'GrLivArea', 'SalePrice')

#plotSalePrice(df)


y = df.loc[:,'SalePrice']

#df = df.drop(['SalePrice'], axis=1)

y = targetLogTrans(y, False)

df, dfTest = processData(df, dfTest)




#df = df[['MSSubClass', 'OverallQual', 'OverallCond', 'GrLivArea', 'GarageArea', 'GarageCars', 'FullBath', 'TotalBsmtSF', '1stFlrSF']]
#dfTest = dfTest[['MSSubClass', 'OverallQual', 'OverallCond', 'GrLivArea', 'GarageArea', 'GarageCars', 'FullBath', 'TotalBsmtSF', '1stFlrSF']]

#pd.set_option('display.max_columns', None)
#print(df)
#plotCorrMatrix(df)

#sys.exit()

X = df.loc[:,df.columns != 'SalePrice']




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)




#0.12892
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=None, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)






gbr = gbr.fit(X,y)

testSetResultsClassifier(gbr, X_test, y_test, 'Gradient boost regression')

saveResult(gbr, dfTest , ids, output='results.csv')



