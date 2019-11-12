import numpy as np
#import matplotlib.pyplot as plt'
import xgboost
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from feature_engine.categorical_encoders import OneHotCategoricalEncoder

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
    p_house(dataset)
    p_workexp(dataset)
    p_satisfaction(dataset)
    p_addIncome(dataset)
#    p_encoding(dataset)
    return dataset


def p_gender(dataset_train):
    dataset_train.Gender = dataset_train.Gender.replace( 'other' ,'missing_gender')
    dataset_train.Gender = dataset_train.Gender.replace( 'f' ,'female')
    dataset_train.Gender = dataset_train.Gender.replace( np.NaN ,'missing_gender') 
    dataset_train.Gender = dataset_train.Gender.replace( 'unknown' ,'missing_gender')
    dataset_train.Gender = dataset_train.Gender.replace( '0' ,'missing_gender')


def p_age(dataset_train):
    age_median = dataset_train['Age'].median()
    dataset_train['Age'].replace(np.nan, age_median, inplace=True)
    #dataset_train['Age'] = (dataset_train['Age'] * dataset_train['Age']) ** (0.5)
    
def p_year(dataset_train):
    #Replacing missing_year year with median
    #p=dataset_train["Year"].mean()
    dataset_train.Year = dataset_train.Year.replace( np.NaN ,dataset_train.Year.median())
    
def p_profession(dataset_train):
    # Transform profession data into categories codes
    dataset_train["Profession"] = dataset_train["Profession"].astype('category')
    dataset_train["profession_cat"] = dataset_train["Profession"].cat.codes
    dataset_train.replace(dataset_train["Profession"],dataset_train["profession_cat"])
    del dataset_train["Profession"]
    dataset_train.profession_cat = dataset_train.profession_cat.replace( np.NaN ,'missing_prof')
    dataset_train.profession_cat = dataset_train.profession_cat.replace( '0' ,'missing_prof')
    
def p_degree(dataset_train):
    #merging University Degree
    dataset_train.Degree = dataset_train.Degree.replace( np.NaN ,'missing_degree')
    dataset_train.Degree = dataset_train.Degree.replace( '0' ,'missing_degree')
    dataset_train.Degree = dataset_train.Degree.replace( 'No' ,'missing_degree')
    
def p_hair(dataset_train):
    #merging Hair Colour
    dataset_train.Hair = dataset_train.Hair.replace( np.NaN ,'Unknown')
    dataset_train.Hair = dataset_train.Hair.replace( '0' ,'Unknown')
    
#size of city
#no changes in size, use directly
    
def p_house(dataset_train):
    #merging Housing Situation
    dataset_train.House = dataset_train.House.replace( np.NaN ,'missing_house')
    dataset_train.House = dataset_train.House.replace( 'nA' ,'missing_house')
  #  dataset_train.House = dataset_train.House.replace( 0 ,'missing_house')
    dataset_train.House = dataset_train.House.replace( '0' ,'missing_house')
    #dataset_train.House = dataset_train.House.replace( 'Medium Apartment','Medium House')
    #dataset_train.House = dataset_train.House.replace( 'Small Apartment' ,'Small House')
    #dataset_train.House = dataset_train.House.replace( 'Large Apartment' ,'Large House')
    ##dataset_train[dataset_train['House'] == '0'] 
   
#crime
#no change use as it is
    
def p_workexp(dataset_train):
    #merging work experience
    dataset_train.WorkExp = dataset_train.WorkExp.replace( '#NUM!', np.NaN)
    dataset_train.WorkExp = dataset_train.WorkExp.replace( np.NaN ,dataset_train.WorkExp.median())
    #the datatype was object so converted to float
    dataset_train['WorkExp'].astype(float)
    dataset_train.WorkExp.dtype                                 
       
def p_satisfaction(dataset_train):                                                
    #merging satis   
    dataset_train.Satisfaction.replace( np.NaN ,'missing_Satis')
     
def p_addIncome(dataset_train):   
    #Extra income to be changed to int from string
    dataset_train.Additional_income = dataset_train.Additional_income.astype(str).str.rstrip(' EUR')
    dataset_train.Additional_income.dtype
    #Now converting this from string to int
    dataset_train['Additional_income'] = dataset_train['Additional_income'].astype(float)

def p_encoding(dataset):
    encoding = ['House','Satisfaction','Gender','Country','Degree','Hair']
    encoder = OneHotCategoricalEncoder(top_categories=None,variables=encoding,drop_last=True)
    encoder.fit(dataset)
    dataset = encoder.transform(dataset)

dataset_train = preprocess(dataset_train)
dataset_predict = preprocess(dataset_predict)

from category_encoders import TargetEncoder
y = dataset_train.Income
X=dataset_train.drop('Income', 1)
#X1 = dataset_predict
t1 = TargetEncoder()
t1.fit(X,y)
X = t1.transform(X)
y1 = dataset_predict.Income
X1=dataset_predict.drop('Income', 1)
X1 = t1.transform(X1)
#TEsting the dataset
#from sklearn.model_selection import train_test_split 
#X=X.drop('Height',1)
#Need to do dummy 

mm_scaler = preprocessing.MinMaxScaler()
X = mm_scaler.fit_transform(X)
X1 = mm_scaler.transform(X1)

#features = fit.transform(X)
## Summarize selected features
#print(features[0:10,:])
#X=pd.get_dummies(X, prefix_sep='_')
#dataset_predict = pd.get_dummies(dataset_predict, prefix_sep='_')
#Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0)

import lightgbm as lgb
X_0 = lgb.Dataset(X, label = y)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
#params['objective'] = 'binary'
#params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 30
params['min_data'] = 50
params['bagging_seed'] = 11
params['max_depth'] = 20

X_1 = lgb.train(params, X_0, 25000)

#from sklearn.linear_model import BayesianRidge
#regressor = BayesianRidge()
##reg = regressor.fit(X, y)
###fitResult = regressor.fit(Xtrain, Ytrain)
##YPred = regressor.predict(dataset_predict)
##from sklearn.ensemble import RandomForestRegressor
##regressor = RandomForestRegressor()
#fitResult = regressor.fit(X, y)
YPred = X_1.predict(X1)
#learningTest = pd.DataFrame({'Predicted': YPredTest, 'Actual': Ytest })
#np.sqrt(metrics.mean_squared_error(Ytest, YPredTest))

#print('Mean Absolute Error:', metrics.mean_absolute_error(Ytest, YPred))
#print('Mean Squared Error:', metrics.mean_squared_error(Ytest, YPred))
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Ytest, YPred)))
data2 = pd.read_csv('tcd-ml-1920-group-income-submission.csv')
data2['Total Yearly Income [EUR]'] = YPred
data2.to_csv('Output.csv', index = False)