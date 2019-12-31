#importing dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn as sns



data = pd.read_csv('')
#print(data.head())
data.head()

g = sns.FacetGrid(data, col="Survived")
g.map(plt.hist, "Age", bins=20)
plt.show()


target_column=['Survived']

train_column=['Age','Pclass','Sex','Fare']

X=data[train_column]
Y=data[target_column]

#data preprocessing 
#check if there is presence of any incompatible data in the train_colum
#print(X['Sex'].isnull().sum())
X['Sex'].isnull().sum()
X['Pclass'].isnull().sum()

#print(X['Age'].isnull().sum())
#find out that there are 177 rows with null value in Age coulmn
#filling it with the median
X['Age']=X['Age'].fillna(X['Age'].median())
print(X['Age'].isnull().sum())

#here sex is either male or female so encoding male as 0 and female as 1

d={'male':0,'female':1}
X['Sex']=X['Sex'].apply(lambda x:d[x])
#print(X['Sex'].head())



#final look in data set
#print(data.head())

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)

#using the SVM

caf=svm.LinearSVC()
caf.fit(X_train,Y_train)
#print the model created
#print(caf)

#testing the model
print (caf.predict(X_test[0:1]))
print (caf.predict(X_test[0:10]))

#printing the accuracy of the model
print(caf.score(X_test,Y_test))

