import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
print(data.columns)
X = [data.Weight]
y = [data.Height]
print(np.shape(X) , np.shape(y))

# mean for X and y
X_mean = np.mean(X)
y_mean = np.mean(y)
print(X_mean , y_mean)

# Find cofficent
M = np.sum((X-X_mean)*(y - y_mean))/np.sum((X - X_mean)**2)
print(M)

#  Find intercepts
C = y_mean - (M*X_mean)
print(C)

# predict
M = int(M)
y_predict = (M*X)+C
print(y_predict)

# Visulization
plt.scatter(X,y)
plt.plot(X,y_predict)
plt.title("Weight vs Height")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()

# How accurate our model is 
R2_score = np.sum((y_predict-y_mean)**2)/np.sum((y-y_mean)**2)
print(R2_score)

print("Our model predicted {:.1f}% accuracy".format(R2_score*100))

# prediction 
x = 92 # We predict height for 92kg weight person
print("92kg weight person has {:.2f} height".format((M*x)+C))
