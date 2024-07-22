# conditional probability : p(A/B) : probability of event A knowing that evnt B has already occured.
#  p(A/B) :[p(B/A) p(A)] / p(B)

# 3 type of Naive Bayes algorithm : Unsupervised learning\Screenshot 2024-06-26 010507.png
# For titanic we can calculate survival rate using p( survived / Male & class & Age & Cabin & fare etc. )
# Naive bayes is generally use for spam mail detection, handwriteen digit recognization, weather prediction, face detection, news artical categorization 

import pandas as pd 
df = pd.read_csv("A:/machine learning/Supervised learning/titanic.csv")
print(df.head())

input = df.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
target = df.Survived

# Age column does not have numerical values. So, we have to convert it using dummy variable or map method 

# inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
#    OR
input = pd.concat([input,pd.get_dummies(input.Sex)],axis='columns')
input = input.drop(['Sex','male'],axis='columns')

print(input.columns[input.isna().any()]) # to check that all data are numerical
print(input.head(15))

input.Age = input.Age.fillna( input.Age.mean() ) 
print(input.head(15))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( input, target, test_size=0.2 , random_state=40)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

print(gnb.score(X_test,y_test))

print(y_test,"/n",gnb.predict(X_test))



# __________ Detect spam emails __________

import pandas as pd 
df = pd.read_csv("A:/machine learning/Unsupervised learning/spam.csv")
#print(df.head())

# In this dataset all data are on text form and machine learning algo work only with numerical data so we have to convert both data into numerical form.

df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df = df.drop('Category',axis='columns')

#print(df.head())

# To convert message into number form we have count vectorizer technique .
# count vectorizer :   
#               (comment it out for results)  
# from sklearn.feature_extraction.text import CountVectorizer
# corpus = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]
# 
# vectorizer = CountVectorizer()       
# x = vectorizer.fit_transform(corpus)  # This will find all unique words from given message and represent that in one line perticular word is present or not using 0 and 1 
# print(vectorizer.get_feature_names_out())    # Return list of unique words
# print(x.toarray())

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
input = vectorizer.fit_transform(df.Message) 
input = input.toarray()

#print(input)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( input, df.spam, test_size=0.2 )

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)

print(mnb.score(X_test,y_test))

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]

emails_count = vectorizer.transform(emails)  # The vectorizer.transform is used to transform new text data based on the vocabulary learned from the training data.
print(mnb.predict(emails_count))



# __________ classify wines into 3 types __________

from sklearn.datasets import load_wine
wine = load_wine()
print(dir(wine))

print(wine.data[:2])
print(wine.target[:2])
print(wine.feature_names)
print(wine.target_names)

import pandas as pd
df = pd.DataFrame(wine.data ,columns=wine.feature_names )
print(df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3 , random_state=150)

from sklearn.naive_bayes import GaussianNB, MultinomialNB , BernoulliNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)
print(gnb.score(X_test,y_test))

mnb = MultinomialNB()
mnb.fit(X_train,y_train)
print(mnb.score(X_test,y_test))
