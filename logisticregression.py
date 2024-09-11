
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
accuracy_score, roc_curve, auc)
np.random.seed(0)
X=np.random.rand(100,2)
y=(X[:,0]+X[:,1]>1).astype(int)
df = pd.DataFrame(np.hstack((X, y.reshape(-1,1))), columns=['Feature1', 'Feature2', 'Target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Accuracy:{accuracy_score(y_test,y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))
print('Classification Report')
print(classification_report(y_test,y_pred))
y_pred_proba = model.predict_proba(X_test)[:,1]
fpr, tpr,_ = roc_curve(y_test,y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr,tpr,color='darkorange', lw=2, label=f'ROC(area={roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Charcteristic(ROC) Curve')
plt.legend(loc="lower right")
plt.show()
