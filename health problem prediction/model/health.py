import pandas as pd
train = pd.read_csv("A:/machine learning/health problem prediction/dataset.csv")
print(train.shape)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_data = label_encoder.fit_transform(train['health problem'])

inputs = train.drop(['health problem','precautions','home remedies'],axis='columns')
target = encoded_data


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

algo = {
    'linear_regression' : {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False] ,
            'positive': [True, False]     
         }
    },
    'random_forest': {
        'model': RandomForestRegressor(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [1,2],
            'selection': ['random', 'cyclic']
            }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['squared_error','friedman_mse'],
            'splitter': ['best','random']
        }
    }     
}
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import ShuffleSplit

scores = []

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for model_name, mp in algo.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=cv, return_train_score=False, scoring='r2')
    clf.fit(inputs, target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
result = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(result)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs , target , test_size=0.3 , random_state=10)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(criterion='squared_error',splitter='best')
model.fit(X_train,y_train)


import numpy as np

def predict_problem(symptoms): 
    print(symptoms)   
    x = np.zeros(len(inputs.columns)) 
    for i in symptoms:
        try: 
            loc_index = np.where(inputs.columns==i)[0][0]  
            if loc_index >= 0:  
                x[loc_index] = 1
        except:
            None
    digit = int(model.predict([x])[0])
    return label_encoder.inverse_transform([digit])[0]

symptoms = []

smp = input("\nenter symptom separated by comma:")
ls = smp.split(',')
symptoms = [item.strip() for item in ls]

print("health problem prediction:",predict_problem(symptoms))

import pickle
with open('Health_problem.pickle','wb') as f:
    pickle.dump(model,f)
import json
sym = {
    'symptoms' : [col.lower() for col in train.columns]
}
with open("symptoms.json","w") as f:
    f.write(json.dumps(sym))

pct = {
    'precautions' : [pct.lower() for pct in train.precautions.unique()]
}
with open("precaution.json","w") as f:
    f.write(json.dumps(pct))

hr = {
    'home_remedies' : [hm.lower() for hm in train['home remedies'].unique()]
}
with open("home_remedies.json","w") as f:
    f.write(json.dumps(hr))


