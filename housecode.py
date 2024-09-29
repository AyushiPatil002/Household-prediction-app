import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv(r'C:\Users\User\Desktop\vs code project\house prediction\House_data.csv')
space=dataset['sqft_living']
price=dataset['price']

x=np.array(space).reshape(-1,1)
y=np.array(price)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)

pred = regressor.predict(xtest)

# Visualize the training set
plt.scatter(xtrain, ytrain, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title('Visuals for Training Dataset')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()

plt.scatter(xtest, ytest, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title('Visuals for Test Dataset')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()


# Check model performance
bias = regressor.score(xtrain, ytrain)
variance = regressor.score(xtest, ytest)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")


# Save the trained model to disk
filename = 'linear_regression_housemodel.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_housemodel.pkl")



import os
print(os.getcwd())