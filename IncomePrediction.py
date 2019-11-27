import numpy as np
#import matplotlib.pyplot as plt'
import pandas as pd
#from sklearn import metrics
from sklearn import preprocessing

#importing dataset
X = pd.read_csv('tcd-ml-1920-group-income-train.csv')
X1 = pd.read_csv('tcd-ml-1920-group-income-test.csv')

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

X.rename(columns=newnames, inplace=True)
X1.rename(columns=newnames, inplace=True)

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

    
#merging Gender
def p_gender(X):
    X["Gender"] = X["Gender"].astype('category')
    X["Gender_cat"] = X["Gender"].cat.codes
    X.replace(X["Gender"],X["Gender_cat"])
    del X["Gender"]
    X.Gender = X.Gender_cat.replace( 'other' ,'missing_gender')
    X.Gender = X.Gender_cat.replace( 'f' ,'female')
    X.Gender = X.Gender_cat.replace( np.NaN ,'missing_gender') 
    X.Gender = X.Gender_cat.replace( 'unknown' ,'missing_gender')
    X.Gender = X.Gender_cat.replace( '0' ,'missing_gender')


def p_age(X):
    age_median = X['Age'].median()
    X['Age'].replace(np.nan, age_median, inplace=True)
    #X['Age'] = (X['Age'] * X['Age']) ** (0.5)
    
def p_year(X):
    #Replacing missing_year year with median
    #p=X["Year"].mean()
    X.Year = X.Year.replace( np.NaN ,X.Year.median())
    
def p_profession(X):
    # Transform profession data into categories codes
    X["Profession"] = X["Profession"].astype('category')
    X["profession_cat"] = X["Profession"].cat.codes
    X.replace(X["Profession"],X["profession_cat"])
    del X["Profession"]
    X.profession_cat = X.profession_cat.replace( '0' ,np.NaN)
    X.profession_cat = X.profession_cat.replace( np.NaN ,"missing_prof")
    
    
def p_degree(X):
    #merging University Degree
    X["Degree"] = X["Degree"].astype('category')
    X["Degree_cat"] = X["Degree"].cat.codes
    X.replace(X["Degree"],X["Degree_cat"])
    del X["Degree"]
    X.Degree_cat = X.Degree_cat.replace( '0' ,np.NaN)
    X.Degree_cat = X.Degree_cat.replace( np.NaN ,"missing_degree")
    
    
def p_hair(X):
    #merging Hair Colour
    X.Hair = X.Hair.replace( '0' ,np.NaN)
    X.Hair = X.Hair.replace( np.NaN ,"missing_hair")

def p_house(X):
    #merging Housing Situation
    X.House = X.House.replace( 'nA' ,np.NaN)
    X.House = X.House.replace( '0' ,np.NaN)
    X.House = X.House.replace( np.NaN ,"missing_house")
    
def p_workexp(X):
    #merging work experience
    X.WorkExp = X.WorkExp.replace( '#NUM!', np.NaN)
    X.WorkExp = X.WorkExp.replace( np.NaN ,X.WorkExp.median())
    #the datatype was object so converted to float
    X['WorkExp'].astype(float)
    X.WorkExp.dtype                                 
       
def p_satisfaction(X): 
                                               
    X.Satisfaction.replace( np.NaN ,'missing_Satis')
     
def p_addIncome(X):   
    #Extra income to be changed to int from string
    X.Additional_income = X.Additional_income.astype(str).str.rstrip(' EUR')
    X.Additional_income.dtype
    #Now converting this from string to int
    X['Additional_income'] = X['Additional_income'].astype(float)


X = preprocess(X)
X1 = preprocess(X1)


from category_encoders import TargetEncoder
y = X.Income
y = y - X['Additional_income']
X = X.drop('Income', 1)
X = X.drop('Instance', 1)
X = X.drop('Additional_income',1)

y1 = X1.Income
y1 = y1 - X1['Additional_income']
X1 = X1.drop('Income', 1)
X1 = X1.drop('Instance', 1)
temp = X1['Additional_income']
X1 = X1.drop('Additional_income',1)

t1 = TargetEncoder()
t1.fit(X,y)
X = t1.transform(X)
X1 = t1.transform(X1)

mm_scaler = preprocessing.MinMaxScaler()
X = mm_scaler.fit_transform(X)
X1 = mm_scaler.transform(X1)

from sklearn.model_selection import train_test_split 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.10, random_state=0)

import lightgbm as lgb
X_0 = lgb.Dataset(Xtrain, label = Ytrain)
X_test1 = lgb.Dataset(Xtest, label = Ytest)

params = {}
params['learning_rate'] = 0.001
params['boosting_type'] = 'gbdt'
params['metric'] = 'mae'
params['verbosity'] = -1
params['bagging_seed'] = 11 
params['max_depth'] = 20


LGB1 = lgb.train(params, X_0, 100000, valid_sets = [X_0,X_test1], early_stopping_rounds=400 )

YPred_lgb = LGB1.predict(Xtest)

final_pred = LGB1.predict(X1)

final_pred = final_pred + temp

data = pd.read_csv('tcd-ml-1920-group-income-submission.csv')
data['Total Yearly Income [EUR]'] = final_pred
data.to_csv('Final.csv', index = False)
