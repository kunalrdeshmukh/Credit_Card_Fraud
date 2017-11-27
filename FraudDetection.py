
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt


# In[2]:

df=pd.read_csv("creditcard.csv")


# In[3]:

df[0:2]


# In[4]:

df.isnull().sum()


# In[5]:

from sklearn.utils import resample
df_majority=df[df.Class==0]
df_minority=df[df.Class==1]
df_minority_upsampled=resample(df_minority,n_samples=284315, random_state=123)
df_upsampled=pd.concat([df_majority,df_minority_upsampled])


# In[6]:

df1 = df_upsampled[['V17', 'V14', 'V12', 'V16', 'V4', 'V11', 'V10', 'V3']].copy()
df2 = df[['V17', 'V14', 'V12', 'V16', 'V4', 'V11', 'V10', 'V3']].copy()


# In[7]:

y=df_upsampled.Class
x=df1
y_real=df.Class
x_real=df2
from sklearn import preprocessing
# x = preprocessing.normalize(x)
# x = preprocessing.scale(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[8]:

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=30)
dtc.fit(x_train,y_train)


# In[9]:

pred=dtc.predict(x_real)


# In[10]:
print("Decision Tree")
from sklearn.metrics import classification_report,confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.metrics import accuracy_score
print("Accuracy= ",accuracy_score(pred,y_real))
print("Classification Report=\n",classification_report(y_real,pred))
tn, fp, fn, tp = confusion_matrix(y_real,pred).ravel()
print("TP=",tp)
print("FP=",fp)
print("FN=",fn)
print("TN=",tn)


# In[11]:

precision_dt, recall_dt, _ = precision_recall_curve(y_real,pred)
FPR_dt,TPR_dt,_ = sk.metrics.roc_curve(y_real,pred)


# In[12]:

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=30)
rfc.fit(x_train,y_train)


# In[13]:

pred_RFC= rfc.predict(x_real)


# In[14]:
print("Random Forest")
print("Accuracy= ",accuracy_score(pred_RFC,y_real))
print("Classification Report=\n",classification_report(y_real,pred_RFC))
tn, fp, fn, tp = confusion_matrix(y_real,pred_RFC).ravel()
print("TP=",tp)
print("FP=",fp)
print("FN=",fn)
print("TN=",tn)


# In[15]:

precision_rf, recall_rf, _ = precision_recall_curve(y_real,pred_RFC)
FPR_rf,TPR_rf,_ = roc_curve(y_real,pred_RFC)


# In[16]:

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(25,20,20,15),max_iter=5000, batch_size=1500)
mlp.fit(x_train,y_train)


# In[17]:

pred_MLP= mlp.predict(x_real)


# In[18]:

print("MLP")
print("Accuracy= ",accuracy_score(pred_MLP,y_real))
print("Classification Report=\n",classification_report(y_real,pred_MLP))
tn, fp, fn, tp = confusion_matrix(y_real,pred_MLP).ravel()
print("TP=",tp)
print("FP=",fp)
print("FN=",fn)
print("TN=",tn)


# In[ ]:

precision_mlp, recall_mlp, _ = precision_recall_curve(y_real,pred_MLP)
FPR_mlp,TPR_mlp,_ = roc_curve(y_real,pred_MLP)


# In[ ]:

from sklearn import svm
svm_classifier = sk.svm.SVC()
svm_classifier.fit(x_train,y_train)


# In[ ]:

pred_SVM = svm_classifier.predict(x_real)


# In[ ]:

print("SVM")
print("Accuracy= ",accuracy_score(pred_SVM,y_real))
print("Classification Report=\n",classification_report(y_real,pred_SVM))
tn, fp, fn, tp = confusion_matrix(y_real,pred_SVM).ravel()
print("TP=",tp)
print("FP=",fp)
print("FN=",fn)
print("TN=",tn)


# In[ ]:

precision_svm, recall_svm, _ = precision_recall_curve(y_real,pred_SVM)
FPR_svm,TPR_svm,_ = roc_curve(y_real,pred_SVM)


# In[ ]:

print "ROC curve"
line_dt, = plt.plot(FPR_dt,TPR_dt,label='Decision Tree')
line_rf, = plt.plot(FPR_rf,TPR_rf,label='Random Forest')
line_mlp, = plt.plot(FPR_mlp,TPR_mlp,label='MLP')
line_svm, = plt.plot(FPR_svm,TPR_svm,label='SVM')
plt.legend(handles=[line_dt, line_rf,line_mlp, line_svm])
plt.title("ROC curve")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:

print "Area under curve: DT,RF,MLP,SVM"
print auc(FPR_dt,TPR_dt),auc(FPR_rf,TPR_rf),auc(FPR_mlp,TPR_mlp),auc(FPR_svm,TPR_svm)


# In[ ]:

print "PR curve"
line_dt, = plt.plot(recall_dt,precision_dt,label='Decision Tree')
line_rf, = plt.plot(recall_rf,precision_rf,label='Random Forest')
line_mlp, = plt.plot(recall_mlp,precision_mlp,label='MLP')
line_svm, = plt.plot(recall_svm,precision_svm,label='SVM')
plt.legend(handles=[line_dt, line_rf,line_mlp, line_svm])
plt.title("PR curve")
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()
print "Area under the curve for Decision Tree: "


# In[ ]:

print "Area under curve: DT,RF,MLP,SVM"
print auc(recall_dt,precision_dt),auc(recall_rf,precision_rf),auc(recall_mlp,precision_mlp),auc(recall_svm,precision_svm)

