import pandas as pd
import streamlit as st 
df = pd.read_csv("new_data.csv")

X = df.iloc[:,0]
Y = df.iloc[:,-1]

#print(X)



from sklearn.feature_extraction.text import CountVectorizer 
count_vect = CountVectorizer()
count_vect.fit(X)
x = count_vect.transform(X)

#print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,Y,test_size = 0.2)

"""from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))"""

from sklearn.neural_network import MLPClassifier
model3 = MLPClassifier()
model3.fit(x_train,y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = model3.predict(x_test)
print(accuracy_score(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))


###FOR INPUT###

import csv

header = ['Words', 'sr.no']
data = ['FUCK', '1']


with open('input.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerow(data)


import pandas as pd
input = pd.read_csv("input.csv")

A = df.iloc[:,0]
B = df.iloc[:,-1]

a = []
count_vect = CountVectorizer()
count_vect.fit(A)
A_num = count_vect.transform(A)
A_out = model3.predict(A_num)
print("Aout=", A_out)
print(len(A_out))

if A_out.any() >= 1:
    print("Not-Bullying")

elif A_out.all() <= 0:
    print("BUllying")



import os
os.remove("input.csv")
#print("File Removed!")
