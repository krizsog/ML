
# coding: utf-8

# In[14]:

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#print("Train shape : ", train.shape)
#print("Test shape : ", test.shape)

PM = [2,10,25,50,75,90]

for prime in PM:
        
    # process columns, apply LabelEncoder to categorical features
    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = LabelEncoder() 
            lbl.fit(list(train[c].values) + list(test[c].values)) 
            train[c] = lbl.transform(list(train[c].values))
            test[c] = lbl.transform(list(test[c].values))

    # shape        
    #print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

    train.y = train.y.astype(int)

    X = train.values[0:, 2:377] # X0 - X385
    Y = train.y.values
    #Y = np.asarray(balance_data['y'], dtype="|S6")

    #X  #values
    #Y #values
    #Let’s split our data into training and test set. We will use sklearn’s train_test_split() method.

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = prime/100, random_state = 100)
    #Decision Tree Classifier with criterion gini index

    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=prime, min_samples_leaf=prime)
    clf_gini.fit(X_train, y_train)

    #Decision Tree Classifier with criterion information gain 

    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=prime, min_samples_leaf=prime)
    clf_entropy.fit(X_train, y_train)

    y_pred = []
    y_pred_en = []

    y_pred = clf_gini.predict(X_test)
    y_pred_en = clf_entropy.predict(X_test)

    print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
    print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
    print(prime)



# In[ ]:



