import numpy as np
#import matplotlib.pyplot as plt'
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

#importing dataset
dataset_train = pd.read_csv('tcd-ml-1920-group-income-train.csv')
dataset_predict = pd.read_csv('tcd-ml-1920-group-income-test.csv')


#Rename columns to not contain spaces
newnames = {"Year of Record" : "Year",
            "Housing Situation" : "House",
          "Crime Level in the City of Employement" : "Crime",
           "Work Experience in Current Job [years]" : "WorkExp",
           "Satisfation with employer" : "Satisfaction",
           "Size of City" : "Size",
           "University Degree" : "Degree",
           "Wears Glasses" : "Glasses",
           "Hair Color" : "Hair",
           "Body Height [cm]" : "Height",
           "Yearly Income in addition to Salary (e.g. Rental Income)" : "Additional_income",
           "Total Yearly Income [EUR]" : "Income"
          }


dataset_train.rename(columns=newnames, inplace=True)
dataset_predict.rename(columns=newnames, inplace=True)


def preprocess(dataset):
    #dataset = dataset[newnames]
    p_gender(dataset)
    p_age(dataset)
    p_year(dataset)
    p_profession(dataset)
    p_degree(dataset)
    p_hair(dataset)
#    p_size(dataset)
    p_crime(dataset)
    p_house(dataset)
    p_workexp(dataset)
    p_satisfaction(dataset)
    p_addIncome(dataset)
#    p_encoding(dataset)
    return dataset

    
#merging Gender
def p_gender(dataset_train):
    #dataset_train.Gender = dataset_train.Gender.replace( 'other' ,'missing_gender')
    dataset_train.Gender = dataset_train.Gender.replace( 'f' ,'female')
    dataset_train.Gender = dataset_train.Gender.fillna(dataset_train.Gender.mode()[0])
    #dataset_train.Gender = dataset_train.Gender.replace( 'unknown' ,'missing_gender')
    #dataset_train.Gender = dataset_train.Gender.replace( '0' ,dataset_train.Gender.mode()[0])

#removing age more than 100
#dataset_train = dataset_train[dataset_train['Age'] <= 100]
def p_age(dataset_train):
    age_median = dataset_train['Age'].median()
    dataset_train['Age'].replace(np.nan, age_median, inplace=True)
#    dataset_train['Age'] = (dataset_train['Age'] * dataset_train['Age']) ** (0.5)
    
def p_year(dataset_train):
    #Replacing missing_year year with median
    #p=dataset_train["Year"].mean()
    dataset_train.Year = dataset_train.Year.replace(np.NaN ,dataset_train.Year.median())
    
def p_profession(dataset_train):
    # Transform profession data into categories codes
#    dataset_train["Profession"] = dataset_train["Profession"].astype('category')
#    dataset_train["profession_cat"] = dataset_train["Profession"].cat.codes
#    dataset_train.replace(dataset_train["Profession"],dataset_train["profession_cat"])
#    del dataset_train["Profession"]
    dataset_train.Profession = dataset_train.Profession.replace(np.NaN ,dataset_train.Profession.mode()[0])
    #dataset_train.profession_cat = dataset_train.profession_cat.replace( '0' ,dataset_train.profession_cat.mode()[0])
    
def p_degree(dataset_train):
    #merging University Degree
    dataset_train.Degree = dataset_train.Degree.replace( np.NaN ,dataset_train.Degree.mode()[0])
    dataset_train.Degree = dataset_train.Degree.replace( '0' ,dataset_train.Degree.mode()[0])
    
def p_hair(dataset_train):
    #merging Hair Colour
    dataset_train.Hair = dataset_train.Hair.replace( np.NaN ,dataset_train.Hair.mode()[0])
    dataset_train.Hair = dataset_train.Hair.replace( '0' ,dataset_train.Hair.mode()[0])

#def p_size(dataset_train):
#    dataset_train['Small City'] = dataset_train.Size <= 2500

def p_house(dataset_train):
    #merging Housing Situation
    dataset_train.House = dataset_train.House.replace( np.NaN ,dataset_train.House.mode()[0])
    #dataset_train.House = dataset_train.House.replace( 'nA' ,dataset_train.House.mode()[0])
    #dataset_train.House = dataset_train.House.replace( '0' ,dataset_train.House.mode()[0])
  
def p_crime(dataset_train):
    dataset_train['No Crime'] = dataset_train.Crime == 0

    
def p_workexp(dataset_train):
    #merging work experience
    dataset_train.WorkExp = dataset_train.WorkExp.replace( '#NUM!', np.NaN)
    dataset_train.WorkExp = dataset_train.WorkExp.replace( np.NaN ,dataset_train.WorkExp.median())
    #the datatype was object so converted to float
    dataset_train['WorkExp'].astype(float)
    dataset_train.WorkExp.dtype                                 
       
def p_satisfaction(dataset_train):                                                
    #merging satis   
    dataset_train.Satisfaction.replace( np.NaN ,dataset_train.Satisfaction.mode()[0])
     
def p_addIncome(dataset_train):   
    #Extra income to be changed to int from string
    dataset_train.Additional_income = dataset_train.Additional_income.astype(str).str.rstrip(' EUR')
    dataset_train.Additional_income.dtype
    #Now converting this from string to int
    dataset_train['Additional_income'] = dataset_train['Additional_income'].astype(float)


dataset_train = preprocess(dataset_train)
dataset_predict = preprocess(dataset_predict)

from category_encoders import TargetEncoder
y = dataset_train.Income
X=dataset_train.drop('Income', 1)
t1 = TargetEncoder()
t1.fit(X,y)
X = t1.transform(X)
y1 = dataset_predict.Income
X1=dataset_predict.drop('Income', 1)
X1 = t1.transform(X1)


#Testing the dataset



mm_scaler = preprocessing.MinMaxScaler()
X = mm_scaler.fit_transform(X)
X1 = mm_scaler.transform(X1)

from sklearn.model_selection import train_test_split 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.12, random_state=0)

#from sklearn.linear_model import BayesianRidge
#regressor = BayesianRidge()
#reg = regressor.fit(X, y)
##fitResult = regressor.fit(Xtrain, Ytrain)
#YPred = regressor.predict(dataset_predict)

#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor()
#fitResult = regressor.fit(Xtrain, Ytrain)
#YPred1 = fitResult.predict(Xtest)
#YPred = fitResult.predict(X1)

import lightgbm as lgb
X_0 = lgb.Dataset(Xtrain, label = Ytrain)
X_test1 = lgb.Dataset(Xtest, label = Ytest)

params = {}
params['learning_rate'] = 0.001
params['boosting_type'] = 'rf'
#params['objective'] = 'binary'
#params['metric'] = 'binary_logloss'
params['metric'] = 'mae'
params['verbosity'] = -1
params['bagging_seed'] = 11 
params['bagging_freq'] = 10
params['bagging_fraction'] = 0.5
params['max_depth'] = 20


X_1 = lgb.train(params, X_0, 100000, valid_sets = [X_0,X_test1], early_stopping_rounds=500 )

YPred1 = X_1.predict(Xtest)
YPred = X_1.predict(X1)

#learningTest = pd.DataFrame({'Predicted': YPredTest, 'Actual': Ytest })
#np.sqrt(metrics.mean_squared_error(Ytest, YPredTest))

print('Mean Absolute Error:', metrics.mean_absolute_error(Ytest, YPred1))
print('Mean Squared Error:', metrics.mean_squared_error(Ytest, YPred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Ytest, YPred1)))

data2 = pd.read_csv('tcd-ml-1920-group-income-submission.csv')
data2['Total Yearly Income [EUR]'] = YPred
data2.to_csv('Output.csv', index = False)
