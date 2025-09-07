import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm # use support vector machine model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # evaluate the model

loan_dataset = pd.read_csv("loan_data.csv")
# print(loan_dataset.shape)
# print(loan_dataset.head())
# print(loan_dataset.describe())
# print(loan_dataset.isnull().sum())

loan_dataset = loan_dataset.dropna()
print(loan_dataset.shape)

loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}}, inplace=True)

# print(loan_dataset.iloc[:,3].value_counts())
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)
print(loan_dataset['Dependents'].value_counts())
# print(loan_dataset['Dependents'].values)

# sns.countplot(x='Education', hue='Loan_Status', data = loan_dataset)
# sns.countplot(x='Married', hue='Loan_Status', data = loan_dataset)
# plt.show()

loan_dataset.replace({'Married':{'Yes':1 , 'No':0},'Gender':{'Male':1,'Female':0},
                      'Self_Employed':{'No':0, 'Yes':1},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}}
                      , inplace=True)
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis = 1)
Y = loan_dataset['Loan_Status']

# print(X)
# print(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1,stratify =Y, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train,Y_train)

#accuracy on training data
X_train_prediction = classifier.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training data:", train_data_accuracy)

#accuracy on testing data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy on testing data:", test_data_accuracy)

