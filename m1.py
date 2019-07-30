import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression


#taking data from quandl
df = quandl.get("EOD/MSFT", authtoken="4WYwYYS5mUJ66Zx6WSCg")

#giving labels
df = df[['Adj_Open','Adj_High','Adj_Low','Adj_Close','Adj_Volume']]

#creating meaningful labels
df['HL_PCT'] = (df['Adj_High']-df['Adj_Close'])/(df['Adj_Close'])*100.0
df['PCT_Change'] = (df['Adj_Close']-df['Adj_Open'])/ df['Adj_Open']*100.0
 
#creating a meaningful output label
df = df[['Adj_Close','HL_PCT','PCT_Change','Adj_Volume']]


#print(df.head())

#done to easily write and change variables
forecast_col = 'Adj_Close'
#to fill the NA places since prediction is tough with NA values
df.fillna(-99999,inplace = True)

#percentage value of shift clockwise or counterclockwise
forecast_out = int(math.ceil(0.01*len(df)))

#shifting the values to the future
df['label']= df[forecast_col].shift(-forecast_out)
df.dropna(inplace= True)
#print(df.tail())

#creating array from labelled columns for X and y
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

#cross validation and splitting up of training set
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size= 0.2)

#which method used for regression -easy to switch algorithm- here linreg is better than svm
#clf = LinearRegression()-96%
#clf = svm.SVR()-94%
#clf = svm.SVR(kernel = 'poly')-56%
clf = svm.SVR(kernel = 'poly')
clf.fit(X_train,y_train)

#parameter which will show how much the training data came near testing data
accuracy = clf.score(X_test,y_test)    
print(accuracy)



 



