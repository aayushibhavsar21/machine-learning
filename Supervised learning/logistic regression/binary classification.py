# Linear Regression: It is used to predict continuous outcomes such as house prices, weather, or stock prices.
# Logistic Regression: It is used to predict categorical outcomes (which have fixed values and the outcome will be from those values), such as whether an email is spam or not, whether a 
#                      customer will buy life insurance, or to which party a person is likely to vote (Democratic, Republican, Independent).                     

# we can not use linear regression line as we used in linear regression because in some cases it gives wrong output ( explanation : https://youtu.be/zM4VZR0px8E?si=ntPl3umm5YhUWX39 )  
# for logistic we can use sigmoid or logit function 
# sigmoid (Z) = 1 / ( 1 + e^-z )   where e = euler's number ~ 2.71828
# here in equation denominator is slightly greater than 1 . So, sigmoid ffunction converts input into range from 0 to 1
# y = 1 / ( 1 + e^(mx + b) )    ; y= mx+b 



#__________ will customer buy life insurance ( binary classification (yes / no ))__________
import pandas as pd 
df = pd.read_csv("A:/machine learning/insurance_data.csv")
print(df)

import matplotlib.pyplot as plt
plt.scatter(df.age,df.bought_insurance)
plt.show()

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.2 , random_state=10)
#print(f"x_train=\n{x_train} , \nx_test=\n{x_test} , \ny_train=\n{y_train} ,\n y_test=\n{y_test}")

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
print( x_test, "\n", model.predict(x_test), "\n", model.score(x_test,y_test))



#__________ will employee leave comapny ( binary classification (yes / no ))__________

import pandas as pd 
df = pd.read_csv("A:/machine learning/HR_comma_sep.csv")

# now we have to analys which data affect employee leaving 

print(df[df.left==1].shape)
print(df[df.left==0].shape)

new_df = df.drop(['Department','salary'],axis='columns')
print( new_df.groupby('left').mean() )

# From result we can draw following conclusions,
# Satisfaction Level : Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)
# Average Monthly Hours : Average monthly hours are higher in employees leaving the firm (199 vs 207)
# Promotion Last 5 Years : Employees who are given promotion are likely to be retained at firm

#Plot bar charts showing impact of employee salaries on retention

import matplotlib.pyplot as plt
pd.crosstab(df.salary,df.left).plot(kind='bar')
plt.show()

pd.crosstab(df.Department,df.left).plot(kind='bar')
plt.show()

# salary also affect the employees , salary have text data .so have to convert into dummy variable  
dummies = pd.get_dummies(df['salary'])
print(dummies)
df = pd.concat([df,dummies],axis='columns')
print(df.head())

x = df.drop(['last_evaluation','number_project','time_spend_company','Work_accident','Department','salary'],axis='columns')
print(x.head())

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split( x , df.left, test_size=0.1 )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)

print(model.predict(x_test),model.score(x_test,y_test))
