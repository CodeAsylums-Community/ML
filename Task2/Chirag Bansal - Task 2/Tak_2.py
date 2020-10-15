from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('kyphosis.csv')

print(df.head())

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))


# Analysing the Data using Numpy

# df = pd.read_csv(r'C:\Users\Chirag Bansal\Desktop\Chirag Bansal\Task 2\kyphosis.csv')

#reading the data from csv file using numpy
ddta = np.genfromtxt(df, delimiter=';' , skip_header=1)
ddta = np.array([ddta])

#printing the shape
print(ddta.shape)

# Total number of patient
print(np.sum(df.Number))

#saving the data in float datatype
print(ddta.astype(np.int))


#modifying the array to a 1D array
print(np.array(df.Kyphosis))

# Average start of kyphosis
print(np.mean(df.Start))

#give dimensions
print(ddta.ndim)

dff = pd.read_csv('glass.csv')

#displaying the shape of glass.csv
print(dff.shape)

# combining the two different csv files horizontaly
dff =np.array(dff.head(81))
combine_data = np.hstack((df, dff))
print(combine_data)

print("******************************************************************")
#combining twwo different data using concate function
print(np.concatenate((df, dff[:,[1,2,3,4]]), axis=0))
