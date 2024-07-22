# This method is use to choose best model for problem with best parameters
# GridSearchCV is use for hyper parameter tuning 

from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target, test_size=0.3)

# Here we are using any random model with random parameters:
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

# We can find score of model more accurately by using k fold technique
# cross_val_score return score for each iteration

from sklearn.model_selection import cross_val_score
print(cross_val_score( SVC(kernel='linear', C=10, gamma='auto'), iris.data, iris.target, cv=5 ) )
print(cross_val_score( SVC(kernel='rbf', C=10, gamma='auto'), iris.data, iris.target, cv=5 ) )
print(cross_val_score( SVC(kernel='rbf', C=20, gamma='auto'), iris.data, iris.target, cv=5 ) )

# OR

import numpy as np
Kernels = ['linear','rbf']
C = [1,10,20]
avg_score = {}

for kval in Kernels:
    for cval in C:
        cv_score = cross_val_score( SVC(kernel=kval, C=cval, gamma='auto'), iris.data, iris.target, cv=5 ) 
        print(f"{kval}-{cval} : {np.average(cv_score)} ")

# For large amount of parameters iteration is not an appropriate method

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV (SVC(gamma='auto'),{
    'C' : [1,10,20] ,
    'kernel' : ['rbf','linear']
} , cv=5 , return_train_score=False)
clf.fit(iris.data,iris.target)
#print(clf.cv_results_)

import pandas as pd
df = pd.DataFrame(clf.cv_results_)

df = df[['param_C','param_kernel','mean_test_score']]

print(clf.best_params_,clf.best_score_,dir(clf))

# different models with different hyperparameters :

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)



# ____________ best model with best parameter for digit dataset ____________

from sklearn.datasets import load_digits
digit = load_digits()

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    'naive_bayes_multinomial': {
        'model': MultinomialNB(),
        'params': {}
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            
        }
    }     
}
from sklearn.model_selection import GridSearchCV
import pandas as pd
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(digit.data, digit.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)
