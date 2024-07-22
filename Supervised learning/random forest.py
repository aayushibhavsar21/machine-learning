#________digit________

from sklearn.datasets import load_digits
digit = load_digits()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digit.data , digit.target , test_size=0.2) 

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=30)    # model use number of random trees defined by estimators to calculate score   
model.fit(X_train,y_train)

print(model.score(X_test,y_test))

# confusion matrix :
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , model.predict(X_test))
print(cm)

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()



#________iris________

from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data , iris.target , test_size=0.2) 

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)    # model use number of random trees defined by estimators to calculate score   
model.fit(X_train,y_train)

print(model.score(X_test,y_test))
