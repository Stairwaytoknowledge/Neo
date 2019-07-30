
#import mean
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style
#import random data
import random

style.use('fivethirtyeight')
    
#xs = [1,2,3,4,5,6]
#ys = [5,4,6,5,6,7]

#converting python list to numpy array

#xs = np.array([1,2,3,4,5,6], dtype = np.float64)
#ys = np.array([5,4,6,5,6,7], dtype = np.float64)




#create a custom dataset 
def create_dataset(hm,variance,step=2, correlation = False):
    
    # function snippets
    #we just begin iterating through the range that we chose with the hm (how much)
    #variable, appending the current value plus a random range of 
    #the negative variance to positive variance.
    #step ->how far to step up y value
    #This gives us data, but currently no correlation if we wanted it.
    #correlation -> if true ,step will be a positive number, if false, it will be neg
    #Let's add that:
    val =1
    # this will be the first value of y before creating random values
    ys  = []
    
    # some range between negative range and positive range of variance 
    #ultimately we get 1- variance or 1 + variance basically
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        # append all those values to ys which will create the random values
    
    # now we have to ask if correlation is true or false, for value to be + or  -
    if correlation and correlation  == 'pos':
        val += step
    elif correlation and correlation == 'neg':
        val -=step
        
    #Great, now we've got a good definition for y values.
    # Next, let's create the xs, which are much easier, 
    #then return the whole thing:
    
    xs = [i for i in range(len(ys))]
        
    return np.array(xs,dtype = np.float64),np.array(ys,dtype = np.float64)
    
    #hm ->the value will be 'how many datapoints we want?'
    #variance ->how much each point will vary from previous, more variance-> less tight data
    #correlation -> can be false,pos or neg to indicate no corr, positive corr or negative corr
    #random -> import random datasets
        


#going to get the best fit slope using a function

#def best_fit_slope(xs,ys):
 #   m = ((mean(xs)*mean(ys) - (mean(xs*ys)))/((mean(xs)*mean(xs))-(mean(xs*xs))))
  #  return m

#m = best_fit_slope(xs,ys)

def best_fit_slope_and_intercept(xs,ys):
    m = ((mean(xs)*mean(ys) - (mean(xs*ys)))/((mean(xs)*mean(xs))-(mean(xs*xs))))
    b = mean(ys)- m*mean(xs)
    return m,b





#FIND OUT THE SQUARED ERROR
        
def squared_error(ys_orig,ys_line):
    return sum((ys_line- ys_orig)**2)







#DEFINE R^2 = 1 - SE(Y^)/SE(Y bar)
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_reg = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - squared_error_reg /squared_error_y_mean



#change how much and variance to adjust

xs,ys = create_dataset(40,40,2,correlation = 'pos')

m,b = best_fit_slope_and_intercept(xs,ys)

#line to identify best fit

regression_line = [(m*x)+b for x in xs]


#let us predict y for different values 
predict_x =8
predict_y = (m*predict_x) + b


r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)



#print(m)


#plt.plot(xs,ys)
plt.scatter(xs,ys)
plt.plot(xs,regression_line)
plt.scatter(predict_x,predict_y, color = 'g')
#step which shows the regression line
#plt.plot(xs, predict_y)

plt.show()
    


