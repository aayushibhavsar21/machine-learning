import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#_________ 1) 1 dependent , 1 independent ( y = mx + c ) _________

# learning from already available data using linear regression model and try to predict value for new data

df = pd.read_csv("A:/machine learning/linear regression/homeprices.csv")

plt.xlabel("area(sqt ft)")
plt.ylabel("price($)")
plt.scatter(df.area , df.price , marker="+")


reg = linear_model.LinearRegression()
print(reg.fit(df[['area']],df.price)) # fitting a data means training linear regression model using available data points.
                                      # reg.fit( df [[ independent variables ]] , target variable )
print( reg.predict([[3300]]) )


#   for any linear equation ( y = mx + c ) we have slope / coefficient (m) and intercept(c) (point where line intersect with y axis) 
#   to predict price for any area model calculate value of slope / coefficient and intercept 

print(reg.coef_ , reg.intercept_)
print( (reg.coef_*3300) + reg.intercept_ ) # mx + c format , output is same as print( reg.predict([[3300]]) )



# predicting prices for csv file of areas and writing back prices to csv file 

area_df = pd.read_csv("A:/machine learning/linear regression/areas.csv")
area_df['prices'] = reg.predict(area_df)
area_df.to_csv("A:/machine learning/linear regression/prediction.csv")



# predicting income for canada per year

new_df = pd.read_csv("A:/machine learning/linear regression/canada_income.csv")

plt.xlabel("year")
plt.ylabel("income in $")
plt.scatter(new_df.year , new_df.income , marker=".")

reg = linear_model.LinearRegression()
reg.fit( new_df[['year']] , new_df.income )
print(reg.predict([[2020]]))

data = {
    'year' : [2017,2018,2019,2020] ,
    'income' : [ reg.predict([[2017]])[0], reg.predict([[2018]])[0], reg.predict([[2019]])[0], reg.predict([[2020]])[0] ]
}

df1 = pd.DataFrame(data)
df1.to_csv("A:/machine learning/linear regression/canada_income.csv" , mode='a' ,index=False ,header=False)



#_________ 2) multiple independent , 1 dependent ( y = m1x1 + m2x2 + m3x3 + ...... + c ) _________

# predicting salary of employee based on experience and score
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from word2number import w2n

df = pd.read_csv("A:/machine learning/linear regression/hiring.csv")

#cleaning data 

df.experience = df.experience.fillna('zero')
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna( math.floor(df['test_score(out of 10)'].mean()))

df.experience = df.experience.apply(w2n.word_to_num)
print(df)


# salary = m1(coefficient 1) * experience + m2(coefficient 2) * test_score + m3(coefficient 3) * interview_score + c(intercept)
# here , we can use linear regression because salary is linearly dependent on other 3 variable 
# i.e. , salary (increase) ~ experience(increase) , test_score(increase) , interview_score(increase)

reg = linear_model.LinearRegression()
reg.fit(df[['experience' ,'test_score(out of 10)' ,'interview_score(out of 10)']] , df['salary($)'])

print(reg.predict([[2 ,9 ,6]]))
print(reg.predict([[12 ,10 ,10]]))


print(reg.coef_,reg.intercept_)
print( (reg.coef_[0] * 2) + (reg.coef_[1] * 9) + (reg.coef_[2] * 6) + reg.intercept_) # output is same as output of print(reg.predict([[2 ,9 ,6]]))
print( (reg.coef_[0] * 12) + (reg.coef_[1] * 10) + (reg.coef_[2] * 10) + reg.intercept_) # output is same as output of print(reg.predict([[12 ,10 ,10]]))



# predicting price of house base on area , bedroom and age

import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('A:/machine learning/linear regression/homevalue.csv')
print(df)

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())

reg = linear_model.LinearRegression()
reg.fit(df.drop('price',axis='columns'),df.price)

print(reg.coef_,reg.intercept_)

print(reg.predict([[3000, 3, 40]]))
print(112.06244194*3000 + 23388.88007794*3 + -3231.71790863*40 + 221323.00186540384 )

print(reg.predict([[2500, 4, 5]]))



# 3)_________ split data set _________

# this method use to split dataset into 2 parts : 
# 1.Training: We will train our model on this dataset
# 2.Testing: We will use this subset to make actual predictions using trained model

# The reason we don't use same training set for testing is because our model has seen those samples before, 
# using same samples for making predictions might give us wrong impression about accuracy of our model. 

import pandas as pd

df = pd.read_csv('A:/machine learning/linear regression/homevalue.csv')

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())

X = df[['area', 'bedrooms', 'age']]
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) 

print(X_train,y_test)

from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
print(clf.score(X_test, y_test))