# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

#Feature Scaling. Required for SVR. Since there's no concept of coefficients
print(y) 
#we need to reshape y because standard scaler class expects a 2D array
y=y.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X= sc_X.fit_transform(X)
# create a new sc object because the first one calcualtes the mean and SD of X. We need different valeus for Y
y= sc_y.fit_transform(y)


print(y)

#Trainig SVR model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#Predicting the result. WE also need to inverse transform to order to get the final result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

#Visualizing the SVR Results
#Inverse scaling in order to plot the original points
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),c='red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)), c='blue') #X is already scaled, so no need to transform in predict
plt.title('Support Vector Regression')
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#Visualizing the SVR results with higher resolution
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()