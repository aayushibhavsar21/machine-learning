# for some complex problems we can not solve it using single lines like linear or logistic regression 

#___________ does salary of employee is >100K ___________
# now we have data which shows in which company and in which department employee get more than 100K salary 
# for this we can make decision tree
# company divides into 3 part google , facebook , ABC pharma this companies are further divides into position of employee and then B tech or M tech

# to know in which order we can divide data like first divide data into company ad then position or first divide into position and then company,
# entropy : we can use entropy ( measure of randomness in sample) . if favorable outcomes are then entropy is low and if favorable outcomes are same as other than entropy is 1 

# gini impurity :

import pandas as pd 

df = pd.read_csv("A:/machine learning/salaries.csv")
print(df.shape)

input = df.drop('salary_more_then_100k',axis='columns')
target = df['salary_more_then_100k']

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

input['company'] = le_company.fit_transform(input['company'])
input['job'] = le_job.fit_transform(input['job'])
input['degree'] = le_degree.fit_transform(input['degree'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input , target , test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

print(model.score(X_test,y_test))

# Is salary of Google, Computer Engineer, Bachelors degree > 100 k ?
print(model.predict([[2,1,0]]))

#Is salary of Google, Computer Engineer, Masters degree > 100 k ?
print(model.predict([[2,1,1]]))



#___________ survived in titanic? ___________

import pandas as pd
df = pd.read_csv("A:/machine learning/titanic.csv") 

input = df.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'],axis='columns')
target = df.Survived 

from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
input['Sex'] = le_sex.fit_transform(input['Sex'])
# or
# input.Sex = input.Sex.map({'male': 1, 'female': 2})

input.Age = input.Age.fillna(input.Age.mean())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input,target,test_size=0.2)
print(len(X_test),len(X_train))

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

print(model.score(X_test,y_test))