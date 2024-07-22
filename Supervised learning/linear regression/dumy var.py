import pandas as pd
df = pd.read_csv("A:/machine learning/linear regression/carprices.csv")

# Typically we use linear regression with quantitative variables. for categorical variable like name, gender, etc. we have to change value 
# of that variable for this we can use dummy variable 

# 1)________ dummy variable ________
# details about dummy variable and dummy var trap --> https://www.statology.org/dummy-variable-trap/ 

dummies = pd.get_dummies(df['Car Model'])
print(dummies)

df = pd.concat([df,dummies],axis='columns')

df = df.drop(['Car Model','Mercedez Benz C class'],axis='columns')
print(df)

from sklearn.linear_model import LinearRegression

model = LinearRegression()  # or from sklearn import linear_model  -> model = linear_model.LinearRegression() 

model.fit(df[['Mileage', 'Age(yrs)' , 'Audi A5' , 'BMW X5']] , df['Sell Price($)'])
#   or
#   X = df.drop('Sell Price($)', axis='columns')
#   y = df['Sell Price($)']
#   model.fit(X,y)

print(model.score(df[['Mileage', 'Age(yrs)' , 'Audi A5' , 'BMW X5']] , df['Sell Price($)']))  # accuracy of model 

print( model.predict( [[45000, 4, 0, 0 ]] ) )
print( model.predict( [[86000, 7, 0, 1 ]] ) )


