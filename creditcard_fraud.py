#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score



#data preprocessing
df = pd.read_csv('./creditcard.csv', sep=',')

total_transaction = len(df)
normal = len(df[df.Class==0])
fraudulent = len(df[df.Class==1])
fraud_percentage = round(fraudulent/normal*100, 2)
print('Total number of Trnsactions are {}'.format(total_transaction))
print('Number of Normal Transactions are {}'.format(normal))
print('Number of fraudulent Transactions are {}'.format(fraudulent))
print('Percentage of fraud Transactions is {}'.format(fraud_percentage))

#dataset information
df.info()
print(min(df.Amount), max(df.Amount))
# Feature Scaling
sc = StandardScaler()
amount = df['Amount'].values
df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))
df.drop(['Time'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)

# Splitting the dataset into the Training set and Test set
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
"""
WE DO NOT HAVE NULL VALUES, BUT IF HAD NULL VALUES SO YOU SHOULD APPLY FEATURE SELECTION MECHANISMS TO
CHECK IF THE RESULTS ARE OPTIMISED, SO UNCOMMENT THE CODE AFTER "TAKING CARE MISSING DATA".

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the K-NN model on the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
knn_accuracy = accuracy_score(y_test, y_pred)
knn_f1_score = f1_score(y_test, y_pred)
print('Accuracy score of the K-Nearest Neighbors model is {}'.format(knn_accuracy))
print('F1 score of the K-Nearest Neighbors model is {}'.format(knn_f1_score))

# Training the Kernel SVM model on the Training set
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
SVM_accuracy = accuracy_score(y_test, y_pred)
SVM_f1_score = f1_score(y_test, y_pred)
print('Accuracy score of the Kernel SVM model is {}'.format(SVM_accuracy))
print('F1 score of the Kernel SVM model is {}'.format(SVM_f1_score))


# Training the Random Forest Classification model on the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
RFC_accuracy = accuracy_score(y_test, y_pred)
RFC_f1_score = f1_score(y_test, y_pred)
print('Accuracy score of the Random Forest model is {}'.format(RFC_accuracy))
print('F1 score of the Random Forest model is {}'.format(RFC_f1_score))


# Training XGBoost on the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
# Making the Confusion Matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
XGBoost_accuracy = accuracy_score(y_test, y_pred)
XGBoost_f1_score = f1_score(y_test, y_pred)
print('Accuracy score of the XGBoost is {}'.format(XGBoost_accuracy))
print('F1 score of the XGBoost is {}'.format(XGBoost_f1_score))

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



#summary

print('Accuracy score of the K-Nearest Neighbors model is {}'.format(knn_accuracy))
print('F1 score of the K-Nearest Neighbors model is {}'.format(knn_f1_score))

print('Accuracy score of the Kernel SVM model is {}'.format(SVM_accuracy))
print('F1 score of the Kernel SVM model is {}'.format(SVM_f1_score))

print('Accuracy score of the Random Forest model is {}'.format(RFC_accuracy))
print('F1 score of the Random Forest model is {}'.format(RFC_f1_score))

print('Accuracy score of the XGBoost is {}'.format(XGBoost_accuracy))
print('F1 score of the XGBoost is {}'.format(XGBoost_f1_score))













