# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:03:46 2020

@author: seans
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')

#chose relevant columns
df.columns

df_model = df[['average_salary','Rating', 'Size', 'Type of ownership','Sector', 
               'Revenue', 'num_comp','job_state', 'same_state', 'age','python_yn', 'spark_yn', 'aws_yn', 'excel_yn','job_simp','seniority', 'desc_len']]

#get dummy data (categorical data)
df_dum = pd.get_dummies(df_model)

#train test split (spliting to train the data)
from sklearn.model_selection import train_test_split

X = df_dum.drop('average_salary', axis =1) #continous variables
y = df_dum.average_salary.values #dependant variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#multiple linear regression
#using stats model ols regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X) #must create a constant for the linear model and a col of 1s does that
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
#model is about 15k off

#lasso regression
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
#broguht it down to 13.629k
#modify the alpha a bit
#running a loop to try differnt alphas and ploting the result
alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

#we see a peak to making the points a tuple, converting to df and finding the max (to find the peak in the plot)
err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]
#peak = aplha- 0.8  error- -13.52k
#plug in new alpha
lm_l = Lasso(alpha=.08)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
#-13.52k error with this model

#random forrest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))
#error = -11.17k ...so much better than previous models due to not worrying about multicollinearity

#tune the model with gridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,60,5), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_

#test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

#testing the models to the y testing data
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf) #the best at 9.04K 

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)

import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

list(X_test.iloc[1,:])
