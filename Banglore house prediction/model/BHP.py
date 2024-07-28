# In this project, we will learn about the kinds of steps and challenges a data scientist goes through for any problem in big companies.
# We will predict the price of property in this project. We will build a website using HTML, CSS, and JavaScript that can predict home prices for us.
# This project includes concepts of data science such as data cleaning, feature engineering, one-hot encoding, outlier detection, dimensionality reduction, and GridSearchCV.

import pandas as pd 
df = pd.read_csv("A:/machine learning/project 1/bengaluru_house_prices.csv")
print(df.head())

# ____________ 1) Data cleaning :  ____________ 

df = df.drop(['area_type','availability','society','balcony'],axis='columns')

print(df.isnull().sum())
df = df.dropna()
print(df.shape)


# ____________ 2) Feature engineering :  ____________ 

print(df['size'].unique())      # The "size" column contains values that represent house size using two different formats: 'n BHK' and 'n Bedrooms'.
df['BHK'] = df['size'].apply(lambda x : int(x.split(' ')[0]))
df = df.drop('size',axis='columns')
print(df.head())

print(df['total_sqft'].unique())   # The "total_sqft" column contains values in a range and in units other than square feet.
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   

df.total_sqft = df.total_sqft.apply(convert_sqft_to_num)
df = df[df.total_sqft.notnull()]

df['price_per_sqft'] = round(df['price']*100000 / df['total_sqft'],2)


# Examine the "locations" variable, which is categorical. We need to apply a dimensionality reduction technique here to reduce the number of locations.

print(len(df['location'].unique()))   # we have 1265 unique location

df.location = df.location.apply(lambda x: x.strip())

location_stats = df['location'].value_counts(ascending=False)

# Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount.
# This method is called Dimensionality Reduction .

df.location = df.location.apply(lambda x : 'other' if x in location_stats[location_stats<=10] else x)
print(len(df.location.unique()))


# ____________ 3) Outlier Removal Using Business Logic :  ____________ 

# Normally square ft per bedroom is 300. If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier.
#  We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft 
print(df.shape)
df = df[~(df['total_sqft']/df.BHK < 300) ]
print(df.shape)

print(df.price_per_sqft.describe())

# Here, we find that the minimum price per sqft is ₹267, while the maximum is ₹12,000,000, indicating a wide variation in property prices.
# The mean represents the average price per sqft, and the standard deviation indicates how much the prices deviate from the mean. 
# Thus, the price can range from the mean minus the standard deviation to the mean plus the standard deviation. Values beyond this range are considered outliers.
# Since prices vary by location, we will first calculate the average price for each location and then remove outliers based on that average.
import numpy as np

def remove_pps_outliers(df):
    df_out = pd.DataFrame()

    for key,subdf in df.groupby('location'):     # here key means location and subdf means dataframe which contain key location only 
        m = np.mean(subdf.price_per_sqft)
        std = np.std(subdf.price_per_sqft)
        reduce = subdf[(subdf.price_per_sqft >= (m-std)) & (subdf.price_per_sqft <= (m+std))]
        df_out = pd.concat( [df_out, reduce],ignore_index=True)

    return df_out

df = remove_pps_outliers(df)

# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like

import matplotlib.pyplot as plt 

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.BHK==2)]
    bhk3 = df[(df.location==location) & (df.BHK==3)]

    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")

    plt.legend()
    plt.show()

plot_scatter_chart(df,"Rajaji Nagar")
plot_scatter_chart(df,"Hebbal")

# We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area).
def remove_bhk_outliers(df):
    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):
        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)

            if stats and stats['count']>5:                
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    
    return df.drop(exclude_indices,axis='index')

# Here, we need to include statistics in the if condition because, if for some locations BHK starts from 3, then rows for 2 BHK (BHK - 1) may not be present in the data. Therefore, we need to check if the weather stats variable is present or not.

df = remove_bhk_outliers(df)
print(df.shape)

plot_scatter_chart(df,"Rajaji Nagar")
plot_scatter_chart(df,"Hebbal")

# if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error 
df = df[df.bath<df.BHK+1]

print(max(df.BHK))
print(max(df.bath))

print(df['location'].nunique())

dummies = pd.get_dummies(df['location'])
df = pd.concat([df,dummies.drop('other',axis='columns')],axis='columns')

df = df.drop(['location','price_per_sqft'],axis='columns')
print(df.shape)

input = df.drop('price',axis='columns')
target = df.price

# classification models like SVC, RandomForestClassifier, LogisticRegression, etc., should only be used for classification tasks.
#    (it is use to predict categorical outcomes ( have fix values , outcome will be only from those values ) like email is spam or not , will customer buy life insurance , to which party person going to vote ( dempcratic , republican , independent ))
# target variable (target) contains continuous values, which are not suitable for a classification model like Naive Bayes. it expects discrete class labels, not continuous values.
# so we will use only 2 models 

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso

algo = {
    'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False] ,
                'positive': [True, False]     
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
    clf.fit(input, target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
result = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(result)
# By using GridSearchCV we can say that linearregression is a best model for our dataset.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.2, random_state=10)
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)

print(input.columns)
print(np.where(input.columns=='1st Block Jayanagar')[0][0])

# function to predict price 
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(input.columns==location)[0][0]       # This line finds the index of the specified location

    x = np.zeros(len(input.columns))             # Initializes a zero array with the same length as the number of columns in dataset i.e.239. 
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:  
        x[loc_index] = 1       # This sets the value of the location feature to 1. All other locations will have their corresponding values in the array set to 0. As we set binary values for location by using dummy variable. 
                               # Here, the value of loc_index is the index we found using the np.where function.
    return lr_clf.predict([x])[0]

print(predict_price('1st Phase JP Nagar',1000, 2, 2))
print(predict_price('1st Phase JP Nagar',1000, 3, 3))
print(predict_price('Indira Nagar',1000, 2, 2))
print(predict_price('Indira Nagar',1000, 2, 3))

# Export the tested model to a pickle file
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

# Export location and column information to a file that will be useful later on in our prediction application
import json
columns = {
    'data_columns' : [col.lower() for col in input.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
