import itertools
import pandas as pd
import numpy as np
import os.path

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Import dataset
pathAbs = os.path.abspath(os.path.dirname(__file__))
pathTraining = os.path.join(pathAbs, 'data-set/train.csv')

df=pd.read_csv(pathTraining)

# Get the shape
df.shape
df.head()

df.loc[(df['label'] == 1) , ['label']] = 'FAKE'
df.loc[(df['label'] == 0) , ['label']] = 'REAL'

# Isolate the labels
labels = df.label
print(labels.head())


#Split the dataset
#TODO aqui divide los datasets, sería bueno probar a usar el set de pruebas test.csv incluso crear uno nuevo con noticias actuales.
x_train,x_test,y_train,y_test=train_test_split(df['text'].values.astype('str'), labels, test_size=0.2, random_state=7)

#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# Fit & transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# Initialize the PassiveAggressiveClassifier and fit training sets
pa_classifier=PassiveAggressiveClassifier(max_iter=50)
pa_classifier.fit(tfidf_train,y_train)

# Predict and calculate accuracy
y_pred=pa_classifier.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f"Accuracy: {round(score*100,2)}")


# Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

