import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error,r2_score # type: ignore
np.random.seed(0)
X = 2*np.random.rand(100,1)
Y = 4+5*X+np.random.rand(100,1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print(f'Coefficient: {model.coef_}')
print(f'Intercept: {model.intercept_}')
print(f'Mean squared error: {mean_squared_error(Y_test,Y_pred)}')
print(f'Coefficient of Determination(R^2):{r2_score(Y_test,Y_pred)}')
plt.scatter(X_test,Y_test,color='black',label='Actual Data')
plt.plot(X_test,Y_pred,color='blue',linewidth=2,label='Regression Line')
plt.title("Simple Linear Regression")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

