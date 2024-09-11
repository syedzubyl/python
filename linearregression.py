import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#sample data creation(replace this with your dataset)
np.random.seed(0)
X = 2*np.random.rand(100, 1)
y = 4+3*X+np.random.randn(100,1)

#convert to dataframe for convenience
df = pd.DataFrame(np.hstack((X,y)), columns=['X', 'y'])

#split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                    random_state=0)

#create linear regression model
model = LinearRegression()

#train the model using the training sets
model.fit(X_train, y_train)

#make predictions using the testing set
y_pred = model.predict(X_test)

#the coefficients
print(f'Coefficients:{model.coef_}')
print(f'Intercept: {model.intercept_}')

#the mean squared error
print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')



#the coefficient of determination:1 is perfect prediction
print(f'Coefficient of determination(R^2):{r2_score(y_test, y_pred)}')

#plot outputs
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=2, label='Regression line')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
      