# multiclass classification : in this type of classification, outcomes will be one of the more than two predefined values.

#__________ recognize digit __________ 

# given exercise is use to recognize handwritten digits. In this we will use training set which lot of handdigit character and 
#   use it to build our logistics model

from sklearn.datasets import load_digits  # -> this contain 1797 image of handwritten digits. We can use it identify our digits
digits = load_digits()                    # -> load training set

print(dir(digits))
# output -> ['DESCR', 'data', 'images', 'target', 'target_names']
#            data   : it is a representation of image as a array with 64(8*8) elements
#            image  : binary representation of digit ( size : 8*8 )
#            target : value of image     
# We will utilize data and target for our model 

#print( digits.data[0:3] )

import matplotlib.pyplot as plt
plt.gray()
for i in range(3):
    #plt.matshow(digits.images[i])
    #plt.show()
    pass

#print(digits.target[0:3])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( digits.data , digits.target , test_size=0.2 )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

print( model.score(X_test,y_test) )

# let's take random number from data set 
#plt.matshow(digits.images[219])
#plt.show()

print( digits.target[219] ) # original value of image

# let's see what our model predict value of that data:
print( model.predict([digits.data[219]]))

# our model accuracy is 0.9374130737134909 so to find that 0.0625869263 where model is not accurate we will use confusion matrix 
y_predict = model.predict( X_test )

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_predict , y_test) # compare both value to find accuracy
print(cm)  # it return in form of array which might difficult to understand so we will use visualization

import seaborn as sn 

plt.figure( figsize = (10,7) )
sn.heatmap(cm , annot=True)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()

# in heatmap number at (0,0) shows that actual and predicated both digits are 0 , same as for (1,1) (2,2).....(9,9) 
# in heatmap number at (1,6) shows actual was 6 but predicated digit was 1 same for others 



#__________ type of iris flower __________ 

from sklearn.datasets import load_iris
iris = load_iris()
print( dir(iris))

#['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']

# DESCR         : A description of the dataset.
# data          : The data array, where each row corresponds to a flower and each column corresponds to a feature (e.g., petal length).
# feature_names : The names of the features (e.g., 'sepal length (cm)').
# filename      : The path to the location of the data file.
# frame         : A pandas DataFrame containing the data and target (if applicable).
# target        : The target array, where each value corresponds to the species of a flower( 0-Setosa , 1-Versicolour , 2-Virginica).
# target_names  : The names of the target classes (e.g., 'setosa', 'versicolor', 'virginica').


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( iris.data , iris.target , test_size=0.2 )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

print( model.score(X_test,y_test) )

print(iris.target[121],iris.target_names[iris.target[121]])
print(iris.target[121],iris.target_names[model.predict([iris.data[121]])])