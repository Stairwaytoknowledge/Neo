
#import mean
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

style.use('fivethirtyeight')
    
#xs = [1,2,3,4,5,6]
#ys = [5,4,6,5,6,7]

#converting python list to numpy array

xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)


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

r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)
    



m,b = best_fit_slope_and_intercept(xs,ys)
#line to identify best fit

regression_line = [(m*x)+b for x in xs]


print(m)

#let us predict y for different values 
predict_x =8
predict_y = (m*predict_x) + b


#plt.plot(xs,ys)
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color = 'g')
#step which shows the regression line
#plt.plot(xs, predict_y)
plt.plot(xs,regression_line)
plt.show()



