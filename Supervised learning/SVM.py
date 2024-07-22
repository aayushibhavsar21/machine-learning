# SVM - support vector machine 
#  |    . . .       -in this graph ,to seperate this 2 data we can draw multiple line from top left to bottom right of graph between cluster of + and .
#  |  +   . . .          but to find best fitted line we can use margin of point with that line . we will use line with maxmimum margin from line to data ponits.
#  | + +     .      
#  |+ +  +         - in 2D space like this grapg boundry between 2 cluster line , in 3D it is a plane and in ND it is hyper plane(not possible to visualize 
#  |____________          but mathematically possible)

# *Support vector machine draws a hyper plane in n dimensional space such that it is maximize margin between classification groups

# gamma - based on how much data region, decision boundry will get decide  
# high gama - in this decision boundary is decided based on only near by data ( margin is checked from only near by data )
#           - This can lead to overfitting . decision boundry might not be straight
# low gama - in this decision boundary is decided based on large data region ( margin is also checked from data which are far away from boundary  )
#          - This can lead to underfitting .
#|
#|     *  *  *   *  *                    
#|   *  *  ** **** * *
#|    *  _______   * *
#|  * * / + +   \   * /  +           --> High gama  
#|   * / +   +   \___/  + +          --> overfitting 
#|           + +    + +  +           --> high regularization(c)
#|           + + ++ + + 
#|______________________________

#|                    /
#|     *  *  *   *  */                    
#|   *  *  ** **** */ *
#|    *            /  * *            --> low gama
#|  * *    + +    /    *   +         --> underfitting 
#|   *   +   +   /        + +        --> low regularization(c)
#|            + /+    + +  + 
#|           + /+ ++ + + 
#|____________/__________________

#__________ SVM model for iris __________
import pandas as pd 
from sklearn.datasets import load_iris
iris = load_iris()

# now we will convert iris data into dataframe to check relation between different flower using length and width of sepal and petal
df = pd.DataFrame(iris.data , columns=iris.feature_names)
print(df.head())

# iris data set has 150 data . 1-49   : target 0 : setosa
#                              50-99  : target 1 :versicolor 
#                              99-150 : target 2 :verginica

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]
import matplotlib.pyplot as plt

# Sepal length vs Sepal Width (Setosa vs Versicolor)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')
plt.show()

#Petal length vs Pepal Width (Setosa vs Versicolor)

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, iris.target, test_size=0.2)

from sklearn.svm import SVC  # svc - support vector clssification 
model = SVC()
model.fit(X_train,y_train)
print(model.score(X_test, y_test))
print(model.predict([[4.8,3.0,1.5,0.3]]))

# Tune parameters :
# 1. Regularization (C)

model_C = SVC(C=1)
model_C.fit(X_train, y_train)
print(model_C.score(X_test, y_test))

model_C = SVC(C=10)
model_C.fit(X_train, y_train)
print(model_C.score(X_test, y_test))

# 2. Gamma

model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
print(model_g.score(X_test, y_test))

#__________ SVM model for digit __________

from sklearn.datasets import load_digits
digit = load_digits() 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digit.data, digit.target , test_size=0.3)

from sklearn.svm import SVC 
model = SVC()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

#Using RBF kernel

rbf_model = SVC(C=1,kernel='rbf')
rbf_model.fit(X_train,y_train)
print(rbf_model.score(X_test,y_test))

linear_model = SVC(kernel='linear')
linear_model.fit(X_train,y_train)
print(linear_model.score(X_test,y_test))