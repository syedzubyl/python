import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix,  classification_report

from sklearn.datasets import load_iris
#load the iris dataset
iris  = load_iris()
X=iris.data
y=iris.target

#for binary classification, we will use only two classes (0 and 1)
X=X[y!=2]
y=y[y!=2]

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
random_state = 42)
#initialize the logistic regression model
model = LogisticRegression()
#train the model
model.fit(X_train, y_train)
#use the trained model to make predictions ont the test data
y_pred = model.predict(X_test)
#accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy:{accuracy:2f}')
#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(conf_matrix)
#classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report')
print(class_report)