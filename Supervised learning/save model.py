# pickel and joblib help us to save model so we can directly used out already trained model and no need to train it again and again 
# because for large data it take time to train model 

# Model which trained to predict house price base on area :

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("A:/machine learning/linear regression/homeprices.csv")

plt.xlabel("area(sqt ft)")
plt.ylabel("price($)")
plt.scatter(df.area , df.price , marker="+")


model = linear_model.LinearRegression()
print(model.fit(df[['area']],df.price)) # fitting a data means training linear regression model using available data points.
                                      # reg.fit( df [[ independent variables ]] , target variable )
print( model.coef_ ,model.intercept_ ,model.predict([[3300]]) )


# 1)______ pickel ______
import pickle

with open('model_pickle','wb') as f:  
    pickle.dump( model ,f )            # save model in current working directory 

with open('model_pickle','rb') as file:
    mp = pickle.load(file)               # load model into memory 

print(mp.coef_ ,mp.intercept_ ,mp.predict([[3300]]))

# 2)______ joblib ______
import joblib 

joblib.dump(model,"model_joblib")   # save model in current working directory 
 
mj = joblib.load('model_joblib')
print(mj.coef_ ,mj.intercept_ ,mj.predict([[3300]]))
