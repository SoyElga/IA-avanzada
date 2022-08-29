import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from implementacion_K_means import *

columns = ["sepal length","sepal width","petal length","petal width", "class"]
df = pd.read_csv('iris.data',names=columns)


X = df[["sepal length","sepal width","petal length","petal width"]].to_numpy()
y = df["class"].factorize()[0]

k = KMeans(K=len(np.unique(y)), max_iters=1000)
y_pred = k.predict(X)

confusion_matrix = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'])

print(confusion_matrix)
print()
print("Accuracy: {acc}%".format(acc = accuracy_score(y, y_pred)*100))