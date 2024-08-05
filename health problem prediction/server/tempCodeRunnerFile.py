

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

import json
import pickle
import numpy as np

__symptoms = None
__precaution = None
__home_remedies = None
__health_problem = None
__model = None
__data = None
__location = None

def load_saved_data():
    global __symptoms 
    global __precaution 
    global __home_remedies 
    global __health_problem
    global __model
    global __data 

    with open("health problem prediction/model/health_problems.json",'r') as f:
        __health_problem = json.load(f)['data_columns']
        label_encoder.fit(__health_problem)       
       
    with open("health problem prediction/model/symptoms.json",'r') as f:
        __data = json.load(f)['symptoms']
        __symptoms = __data[:-3]
    
    with open("health problem prediction/model/precaution.json",'r') as f:
        __precaution = json.load(f)['precautions']

    with open("health problem prediction/model/home_remedies.json",'r') as f:
        __home_remedies = json.load(f)['home_remedies']

    with open("A:/machine learning/health problem prediction/model/Health_problem.pickle",'rb') as f:
        __model = pickle.load(f)

def predict_problem(sym):
    print(sym)   
    global __location 
    x = np.zeros(len(__symptoms)) 
    for i in sym:
        try: 
            loc_index = __symptoms.index(i.lower())  
            if loc_index >= 0:  
                x[loc_index] = 1
        except:
            None
    digit = int(__model.predict([x])[0])
    hp = label_encoder.inverse_transform([digit])[0] 
    __location = __health_problem.index(hp)
    return hp 

def get_precaution():
    if __location is not None:
        return __precaution[__location]

def get_home_remedies():
    if __location is not None:
        return __home_remedies[__location]

def get_symptoms():
    return __symptoms

if __name__ == '__main__':
    load_saved_data()
    s = input("\nenter symptom separated by comma:")
    ls = s.split(',')
    sym = [item.strip() for item in ls]
    print("health problem prediction:",predict_problem(sym))
