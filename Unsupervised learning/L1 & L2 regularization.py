# L1 and L2 regularization generally use for overfiting
#  equation for underfitting is linear
#      ( eg : 0o + 01*age )
#  equation for overfitting is high order polynomial 
#      ( eg : 0o + 01*age +  02*age^2 +  03*age^3 +  04*age^4 )
#  equation for Balancedfitting is polynomial 
#      ( eg : 0o + 01*age +  02*age^2 )

# Overfitting does not generalize really well and can not predict for new data accurately.
# If we have condition of overfitting and somehow we make sure that 03 and 04 in equation for 
#     overfitting, is close to zero than we can convert it into balanced fit. 

# Mean squared error = 1/n sigma( Y(original)i -Y(predicated)i )^2 i from 1 to n  ; mse is use to find bestfit of graph
#      where y = 0o + 01*X +  02*X^2 +  03*X^3 +  04*X^4       
#                MSE = 1/n sigma( Y(original) -Y(predicated) )^2 + lambda* sigma (0i^2) i from 1 to n
# here last part of equation ensure that value of 0 does not increse to much
# if value of 0 is high then MSE is also increase , it mean that line for which we are calculating MSE is not best fit 
# This is how it ensure that value of 0 stay near to zero

#  L1 regularization : MSE = 1/n sigma( Y(original) -Y(predicated) )^2 + lambda* sigma |0i| i from 1 to n
#  L2 regularization : MSE = 1/n sigma( Y(original) -Y(predicated) )^2 + lambda* sigma (0i^2) i from 1 to n


import pandas as pd 
df = pd.read_csv("A:/machine learning/Unsupervised learning/Melbourne_housing_FULL.csv")

print(df.nunique())

df = df[['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']]

# Some feature's missing values can be treated as zero (another class for NA values or absence of that feature)
# like 0 for Propertycount, Bedroom2 will refer to other class of NA values
# like 0 for Car feature will mean that there's no car parking feature with house
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
df [cols_to_fill_zero] = df [cols_to_fill_zero].fillna(0)

# other continuous features can be imputed with mean for faster results since our focus is on Reducing overfitting
# using Lasso and Ridge Regression
df ['Landsize'] = df ['Landsize'].fillna(df .Landsize.mean())
df ['BuildingArea'] = df ['BuildingArea'].fillna(df .BuildingArea.mean())

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Price', axis=1)
y = df['Price']

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train_X, train_y)

print(reg.score(test_X, test_y))
print(reg.score(train_X, train_y))

# Normal Regression is clearly overfitting the data

# Using Lasso (L1 Regularized) Regression Model:
from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)

print(lasso_reg.score(test_X, test_y))
print(lasso_reg.score(train_X, train_y))

# Using Ridge (L2 Regularized) Regression Model
from sklearn.linear_model import Ridge
ridge_reg= Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)

print(ridge_reg.score(test_X, test_y))
print(ridge_reg.score(train_X, train_y))

