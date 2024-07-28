# Cross-validation is a technique used to determine which machine learning model (SVM, logistic regression, or random forest) will be most accurate for a given problem.
# It helps to assess the accuracy of all the models provided for the problem.
# For more accurate results, we typically divide the data into two parts: training and testing. However, it’s possible that the data contains different categories, and the training part might contain one category while the testing part contains another.
# In such cases, accuracy may be compromised. To address this, we use the k-fold cross-validation technique. This technique divides the data into a specified number of folds and evaluates the model by using each fold as the test set while the remaining folds serve as the training set.
# _________ best algorithm for digit dataset _________

from sklearn.datasets import load_digits
digit = load_digits()

from sklearn.model_selection import KFold
kf = KFold(n_splits=3)


# example of how this method fold data into k parts:
# Here , k fold divide data into 3 parts , 1 is for training and other 2 is for testing 
# It will change its testing data set and will take all fold as testing data one by one 

for train_index , test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index , test_index) 


# working of Kfold for any problem to find best model:

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

scores_logistic = []
scores_svm = []
scores_rf = []


# stratifiedkfold is same as kfold but it divide each classification categories in uniform way

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

for train_index, test_index in folds.split( digit.data , digit.target ):
    X_train, X_test, y_train, y_test = digit.data[train_index] , digit.data[test_index] , digit.target[train_index] , digit.target[test_index]

    scores_logistic.append( get_score( LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))  
    scores_svm.append( get_score( SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append( get_score( RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

import numpy as np 

print(scores_logistic , np.mean(scores_logistic))
print(scores_svm , np.mean(scores_svm))
print(scores_rf , np.mean(scores_rf))

# we can conclude that random forest is best algo for digit dataset 

# we can directly get score of model for problem . There is not any need to difine get_score function and find score of every models like we did . ( It was just for explanation about how this method work )
# cross_val_score function

from sklearn.model_selection import cross_val_score

# Logistic regression model performance using cross_val_score
print(cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digit.data, digit.target,cv=3))  # this output is same as array (score_logistic) we have declared for different scores of models

# svm model performance using cross_val_score
print(cross_val_score(SVC(gamma='auto'), digit.data, digit.target,cv=3))        # this output is same as array (score_SVC) we have declared for different scores of models

# random forest performance using cross_val_score
print(cross_val_score(RandomForestClassifier(n_estimators=40),digit.data, digit.target,cv=3))   # this output is same as array (score_rf) we have declared for different scores of models

# cross_val_score uses stratifield kfold by default

# Parameter tunning using k fold cross validation
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),digit.data, digit.target, cv=10)
print(np.average(scores1))

scores2 = cross_val_score(RandomForestClassifier(n_estimators=20),digit.data, digit.target, cv=10)
print(np.average(scores2))

scores4 = cross_val_score(RandomForestClassifier(n_estimators=40),digit.data, digit.target, cv=10)
print(np.average(scores4))

# Here we used cross_val_score to fine tune our random forest classifier and figured that having around 40 trees in random forest gives best result.



# _________ best algorithm for iris dataset _________

from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np

print( np.average(cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), iris.data, iris.target,cv=3)) )  

print( np.average(cross_val_score(SVC(gamma='auto'), iris.data, iris.target,cv=3)) )       

print( np.average(cross_val_score(RandomForestClassifier(n_estimators=40),iris.data, iris.target,cv=3)) )   

print( np.average(cross_val_score(DecisionTreeClassifier(),iris.data, iris.target,cv=3)) )   
