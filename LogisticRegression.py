import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris
iris = load_iris()
X=iris.data
y=iris.target
X=X[y!=5]
y=y[y!=5]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy:{accuracy:.2f}')
conf_matrix=confusion_matrix(y_test,y_pred)
print('Confusion matrix:')
print(conf_matrix)
class_report=classification_report(y_test,y_pred)
print('classification Report:')
print(class_report)
