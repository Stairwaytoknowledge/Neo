
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

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
        for features in data[group]:
            #hard code for two dimensions
            #euclidian_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2)
            euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
            #this is much faster than the equivalent np.sqrt((np.array(features)-np.array(predict))**2)
            distances.append([euclidian_distance,group])
            #we need to append the euclidian_distance and group to sort that list
            
    #sort the votes         
    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbors(dataset,new_features,k =3)
print(result)

[[plt.scatter(ii[0],ii[1],s=100,color = i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],color = result)   
plt.show()