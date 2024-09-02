import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

np.random.seed(0)
X1=np.random.rand(100,1)
X2=np.random.rand(100,1)
Y=3*X1+6*X2+np.random.rand(100,1)

X=np.hstack((X1,X2))
df=pd.DataFrame(np.hstack((X,Y)),columns=['X1','X2','Y'])

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

model = LinearRegression()

model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

print(f'Coefficient: {model.coef_}')
print(f'Intercept: {model.intercept_}')
print(f'Mean squared error: {mean_squared_error(Y_test,Y_pred)}')
print(f'Coefficient of Determination(R^2):{r2_score(Y_test,Y_pred)}')

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.scatter(X_test[:,0],Y_test,color='black',label='Actual Data')
plt.scatter(X_test[:,0],Y_pred,color='blue',linewidth=1,label='Predicted Data')
plt.title("X1 vs Y")
plt.xlabel('X1')
plt.ylabel('Y')
plt.legend()

plt.subplot(1,2,2)
plt.scatter(X_test[:,1],Y_test,color='black',label='Actual Data')
plt.scatter(X_test[:,1],Y_pred,color='blue',linewidth=1,label='Predicted Data')
plt.title("X2 vs Y")
plt.xlabel('X2')
plt.ylabel('Y')
plt.legend()


plt.tight_layout()
plt.show()