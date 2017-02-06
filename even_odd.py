from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import random

def int_to_binary(value):
    return list("{:#018b}".format(value))[2:]

iris_dataset = load_iris()
knn = KNeighborsClassifier(n_neighbors=1)

COUNT = 9999
values = np.zeros((COUNT, 16))
answers = np.zeros(COUNT, dtype=int)

for i in range(0,COUNT):
    v = int_to_binary(i+1)
    values[i] = np.asarray(v)

for i in range(0,COUNT):
    if((i+1) % 2 == 0):
        answers[i] = 1
    else:
        answers[i] = 0

#print(answers)

#print(X_test)
#print(knn.predict(X_test))

X_train, X_test, y_train, y_test = train_test_split(
    values,
    answers

)

#print(X_train.shape)
#print(y_train.shape)

knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

brand_new = []
brand_new_int = []
for i in range(0,20):
    v = random.randint(1, 9999)
    brand_new_int.append(v)
    brand_new.append(int_to_binary(v))


predictions = knn.predict(brand_new)
for i in range(0, len(brand_new)):
    print("{} = {} = {}".format(brand_new_int[i], predictions[i], (brand_new_int[i] % 2) == 0))
