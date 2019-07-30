import numpy as np
from math import sqrt
#import matplotlib.pyplot as plt
import warnings
#from matplotlib import style
from collections import Counter
import pandas as pd
import random
import pdb

#style.use('fivethirtyeight')

#dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
#new_features = [5,7]

#one line for loop to scatter
#for i in dataset:
#for ii in dataset[i]:
#[[plt.scatter(ii[0],ii[1],s=100,color = i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0],new_features[1])   
#plt.show()


#define a function, we need to pass training data, we need to predict and we need a value of k
def k_nearest_neighbors(data,predict,k=3):
    if len(data)>=k :
        warnings.warn('K is set to a value less than total voting groups')
# we have to compare our point to all  other points
# we can chose to compare in a fixed radius and avoid the outliers
# k in nearest neigbors means the number of nearest neighbors from which we wish to take votes
#for example if k =3 , then we compare our point with nearest 3 points
    distances = []
    for group in data:
        # to get the groups in dataset passed
        for features in data[group]:
            #to get individual features in each group
            #hard code for two dimensions
            #euclidian_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2)
            euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
            #this is much faster than the equivalent np.sqrt((np.array(features)-np.array(predict))**2)
            #linear algebra norm
            distances.append([euclidian_distance,group])
            #we need to append the euclidian_distance and group to sort that list
            #list of list, the distance and group are listed
            
    #sort the votes         
    votes = [i[1] for i in sorted(distances)[:k]]
    # equivalent to for i in sorted(distances)[:k]
    print(votes)
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    print('vote result is',vote_result)
    return vote_result

#result = k_nearest_neighbors(dataset,new_features,k =3)
#print(result)

#[[plt.scatter(ii[0],ii[1],s=100,color = i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0],new_features[1],color = result)   
#plt.show()

df= pd.read_csv('breast-cancer-wisconsin.data')
df.columns =["id", "clump_thickness", "unif_cell_size", "unif_cell_shape", "marg_adhesion",
"single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"]
df.replace('?',-99999,inplace = True)
df.drop(['id'], 1, inplace = True)
full_data = df.astype(int).values.tolist()

#print(df.head())
#shuffle the data since we have converted this to a list of list
random.shuffle(full_data)
#print(20*'#')
print(full_data[:10])


#split to test set and training set
#TEST SET
test_size = 0.2
# final 20 percent of shuffled data

#Now we populate the dictionaries.
# If it is not clear, the dictionaries have two keys: 2 and 4.
# The 2 is for the benign tumors (the same value the actual dataset used), 
#and the 4 is for malignant tumors, same as the data. 
#We're hard coding this, but one could take the classification column, 
#and create a dictionary like this with keys that were assigned by unique column
# values from the class column. We're just going to keep it simple for now, however.

train_set = {2:[],4:[]}
test_set = {2:[],4:[]}

#train data->multiply it by the test_size* full data length to slice up the data to 80 20
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]


# the last twenty percent of the data
#populating dictionaries
#key is the class, values are the atrributes
#print(type(train_data))
for i in train_data:
    train_set[i[-1]].append(i[:-1])
#-1 indicate the last value -> class 
# we are appending list to this list -> elements upto the last element except the class

#print(type(test_data))
for i in test_data:
    test_data[:i[-1]].append(i[:-1])  
    

#training and testing    
correct = 0
total = 0
print(' correct and total are ',correct,total)

for group in test_set:
        #for each 2 and 4
   for data in test_set[group]:  
        #for each data in test-set[group]
        # the data we are going to pass through predict(test_set) and data(data)
        vote = k_nearest_neighbors(train_set,data, k =5)
        # if the group that came is equal to the vote
        if group ==vote:
            correct +=1
        total +=1
        print('Accuracy is ', float(correct//total))
