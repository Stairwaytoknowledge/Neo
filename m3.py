# going to predict wide into the future

import pandas as pd
import quandl
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt

# import pickle to save the training and use it for further analysis everytime
import pickle


from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
from matplotlib import style

style.use('ggplot')

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
#Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj_Close'
#to fill the NA places since prediction is tough with NA values
#drop missing value
df.fillna(-99999,inplace = True)

#percentage value of shift clockwise or counterclockwise
#We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01*len(df)))

#shifting the values to the future

df['label']= df[forecast_col].shift(-forecast_out)
#df.dropna(inplace= True)
#print(df.tail())

#creating array from labelled columns for X and y
X = np.array(df.drop(['label'],1))
#Scale the X so that everyone can have the same distribution for linear regression
#No preprocesssing to avoid data being held up for days
X = preprocessing.scale(X)

#Remove forecast out values 
X = X[:-forecast_out]
# Latest data against which prediction is done
# Finally We want to find Data Series of late X and early X (train)
# for model generation and evaluation> WE DONT HAVE y VALUES OF THOSE X
X_lately=X[-forecast_out:]

# To remove NA values
df.dropna(inplace= True)


y = np.array(df['label'])
#y = np.array(df['label'])

#cross validation and splitting up of training set
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size= 0.2)

#which method used for regression -easy to switch algorithm- here linreg is better than svm
#EVERY TIME YOU NEED TO PREDICT YOU TRAIN
clf = LinearRegression(n_jobs = -1)
#clf = svm.SVR()-94%
#clf = svm.SVR(kernel = 'poly')-56%
#clf = svm.SVR(kernel = 'poly')
clf.fit(X_train,y_train)

# HERE WE ARE USING PICKLE
#EVERY TIME IT IS TRAINED SAVE IT so that repeated attempts are not necessary
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)

    
pickle_in = open('linearregression.pickle','rb')
#LOADING PICKLE
clf = pickle.load(pickle_in)

#TO CHECK WHETHER IT WORKED COMMENT THE 3 LINES ABOVE PICKLE_IN and we get the 
#SAME EXACT RESULT


#parameter which will show how much the training data came near testing data
accuracy = clf.score(X_test,y_test)    

#try to forecast y, pass single or array of values in clf.predict
#this is the crux of the prediction with  scikit learn
#we can send a single value or an array of value and it will just output the 
#array of values
forecast_set = clf.predict(X_lately)


#stock up unknown 30 days value
print(forecast_set,accuracy, forecast_out)

df['Forecast'] = np.nan
#last name  
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()

one_day =86400
next_unix = last_unix + one_day

#y is the feature, X is not, so we have to enter date values
# we have to populate the data frame with the new date values
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    #one liner for loop for removing nan across the df.columns
    #this shows that the next date is date stamp. we need to index it 
    # then using the for loop we are saying that all columns in the current df
    # are non nan 
    #In this entire for loop we have tried to create an index using date 
    # i in forecast is NAN is added at the end of the column


#Observe that in this step we dont have values for forecast but have values for every other step
print(df.head())

# Observe that in this case we dont have values for future ,but we do have a forecast
print(df.tail())

#plotting using matplotlib 
df['Adj_Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



