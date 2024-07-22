import json
import pickle
import numpy as np

__location = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())       # This line finds the index of the specified location
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))             # Initializes a zero array with the same length as the number of columns in dataset i.e.239. 
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:  
        x[loc_index] = 1       # This sets the value of the location feature to 1. All other locations will have their corresponding values in the array set to 0. As we set binary values for location by using dummy variable. 
                               # here value of loc_index is value we have found by using np.where function .  
    return round(__model.predict([x])[0],2)

def get_location_names():
    return __location

def load_saved_data():
    print('loading saved data....start')
    global __data_columns
    global __location
    global __model

    with open("Banglore house prediction/model/columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __location = __data_columns[4:]

    with open("Banglore house prediction/model/banglore_home_prices_model.pickle",'rb') as f:
        __model = pickle.load(f)

    print('loading saved data....done')

if __name__ == '__main__':
    load_saved_data()
    #print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 2, 2))
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar',2000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar',2000, 4, 4))
    print(get_estimated_price('1st Phase JP Nagar',2000, 5, 5))
    print(get_estimated_price('Khalhalli',1000, 2, 2))
    print(get_estimated_price('ejipura',1000, 2, 3))