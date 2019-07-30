
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd


df = pd.read_csv('breast-cancer-wisconsin.data')
# algorithm will treat this as an outlier if this step is done
# columns 
df.columns =["id", "clump_thickness", "unif_cell_size", "unif_cell_shape", "marg_adhesion",
"single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"]

df.replace('?',-99999, inplace = True)

#we need to drop ID since it will drastically distort k neighbor, check by commenting
df.drop(['id'], 1, inplace = True)

#features - everything except the class
X = np.array(df.drop(['class'],1))
#labels- just the class
y = np.array(df['class'])

#Training and testing 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#test size is the percentage of data meant for training and testing


# create k neighbor classifier
clf = neighbors.KNeighborsClassifier()

# fit it with random training data
clf.fit(X_train,y_train)

#predicting accuracy
accuracy =  clf.score(X_test,y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
#example_measures = example_measures.reshape(2,-1)
#instead we can use len of example measures while reshaping
example_measures = example_measures.reshape(len(example_measures),-1)
# It simply means that it is an unknown dimension and we want numpy to figure it out. 
#And numpy will figure this by looking at the 'length of the array
# and remaining dimensions' and making sure it satisfies the above mentioned criteria
prediction = clf.predict(example_measures)

print(prediction)

