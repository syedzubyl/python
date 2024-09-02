import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_curve,auc
np.random.seed(0)
x = np.random.rand(100,2)
y = (x[:,0]+x[:,1]>1).astype(int)
df = pd.DataFrame(np.hstack((x,y.reshape(-1,1))),columns = ['Feature1','Feature2','Target'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(f'Accuracy:{accuracy_score(y_test,y_pred)}')
print('Confusion matrix')
print(confusion_matrix(y_test,y_pred))
print('Classification Report:')
print(classification_report(y_test,y_pred))
y_pred_proba = model.predict_proba(x_test)[:,1] 
fpr,tpr,_= roc_curve(y_test,y_pred_proba)
roc_auc=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='orange',lw=2,label=f'ROC curve(area={roc_auc:.2f})')
plt.plot([0,1],[0,1],color='blue',lw=2,linestyle='--')
plt.xlim([0.0,1.0])
plt.xlim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Charactersitic(ROC)curve')
plt.legend(loc="lower right")
plt.show()