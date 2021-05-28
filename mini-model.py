#importing essential libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import plotly.express as px
import plotly.io as pio
warnings.filterwarnings("ignore")

df= pd.read_csv("heart.csv")

df.head()

df.describe()

df.isnull().sum()

plt.figure(figsize= (15,6))
sns.set_style("darkgrid")
sns.heatmap(df.corr(),annot= True)
plt.show()

df.var()

#Using log transformation
df["age"]= np.log(df.age)


df["trtbps"]= np.log(df.trtbps)
df["chol"]= np.log(df.chol)
df["thalachh"]= np.log(df.thalachh)

df.var()

df.describe()

#importing essential libraries
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#train test split
label= df["output"]
train= df.drop("output",axis= 1)

x_train,x_test,y_train,y_test= train_test_split(train,label,test_size= 0.25,random_state= 5)

knn= KNeighborsClassifier(n_neighbors= 13)
knn.fit(x_train,y_train)
knnpred = knn.predict(x_test)
accuracy_score(y_test,knnpred)

#confusion matrix
cm= confusion_matrix(y_test,knnpred)
sns.heatmap(cm,annot= True)

#classification report
cr= classification_report(y_test,knnpred)
cr

import pickle

# Saving model to disk
pickle.dump(knn, open('miniproject.pkl','wb'))

model = pickle.load(open('miniproject.pkl','rb'))

knnpred = model.predict(x_test)
accuracy_score(y_test,knnpred)


