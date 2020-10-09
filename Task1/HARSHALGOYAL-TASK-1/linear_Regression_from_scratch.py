
#importing the necessary modules
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#calcuating the mean
def mean(values):
	return sum(values)/float(len(values))


#calculating the variance
def variance(values,mean):
	return sum([(x-mean)**2 for x in values])


#calcuating the coeffcients b1 and b0 for the equation yhat = b0+b1*x
def coefficients(iris_X,mean_iris_X,iris_Y,mean_iris_Y,var_iris_X):

	n = len(iris_X)
	numerator = 0

	for i in range(n):
		numerator += (iris_X[i] - mean_iris_X) * (iris_Y[i] - mean_iris_Y)
	
	b1 = numerator/var_iris_X
	
	b0 = mean_iris_Y - (b1 * mean_iris_X)
	
	return(b1,b0)

	
#training model and predecting values of dependent variables
def linear_regression(iris_X_train,iris_Y_train,iris_X_test,iris_Y_test):
	predection =  []
	
	mean_iris_X , mean_iris_Y = mean(iris_X_train), mean(iris_Y_train)

	var_iris_X , var_iris_Y = variance(iris_X_train,mean_iris_X), variance(iris_Y_train,mean_iris_Y)

	for x in iris_X_test:
		b1,b0=coefficients(iris_X_train,mean_iris_X,iris_Y_train,mean_iris_Y,var_iris_X)
		yhat = b0 + b1*x
		predection.append(yhat)
	return predection


#calcuting the mean squared error
def calc_error(predected,iris_Y_test):
	m = len(predected)
	error = 0
	for ele in predected:
		i = 0
		error += ((ele- iris_Y_test[i])**2 / m)
		i+=1
	return error





#main function
iris_X , iris_Y = datasets.load_iris(return_X_y = True) #loading the dataset from the sklearn
iris_X = iris_X[:,np.newaxis,2] 						#settings only petal length and petal width as features 




iris_X_train,iris_X_test,iris_Y_train,iris_Y_test = train_test_split(iris_X,
																	  iris_Y,
																	  test_size=.32,random_state=30)	#splitting the data into test and train datset


predected = linear_regression(iris_X_train,iris_Y_train,iris_X_test,iris_Y_test)	#calling the linear_regression function to predict the values of y

print("Mean squared error : ", calc_error(predected,iris_Y_test))	#displaying the error 


#visulaising the model
plt.scatter(iris_X_test,iris_Y_test, color = 'red')
plt.plot(iris_X_test,predected, color = 'blue',linewidth = 2)

#displaying the plot on screen
plt.show()