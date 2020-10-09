
## importing necessary modules
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


iris_X , iris_Y = datasets.load_iris(return_X_y = True) #loading the iris dataset and storing the dependent variable in iris_X and independent variable in Y

iris_X = iris_X[:,np.newaxis,2]	#taking only the length and breadth of petals as features for training


iris_X_train,iris_X_test,iris_Y_train,iris_Y_test = train_test_split(iris_X,
																	  iris_Y,
																	  test_size=.32,random_state=30)	# splitting the dataset into test and train dataset with a test size of 32% of the dataset

regressor = LinearRegression()	#creating the instance of LinearRgression class

regressor.fit(iris_X_train,iris_Y_train) 	#training the model on the given features and labels


predict_y = regressor.predict(iris_X_test)	#predicting the labels(dependent variable values) for the test features(X)



print('Predection accuracy : %.2f' %regressor.score(iris_X_test,iris_Y_test))	#score for r2 testing
print('Mean Squared error is : %.2f'% mean_squared_error(iris_Y_test,predict_y))	#claculating the mean squared error for the model

#visualising the model acuuracy
plt.scatter(iris_X_test,iris_Y_test, color = 'red')
plt.plot(iris_X_test,predict_y, color = 'blue',linewidth = 2)

#displaying the plot on screen
plt.show()
