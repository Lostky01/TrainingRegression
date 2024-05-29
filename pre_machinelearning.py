# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(0)

x = np.random.rand(100,2)
y = (x[:,0] + x[:,1] > 1).astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LogisticRegression()

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
class_report = classification_report(Y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)