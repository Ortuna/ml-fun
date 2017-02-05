from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris_dataset = load_iris()
knn = KNeighborsClassifier(n_neighbors=1)

values = list()
answers = list()
for i in range(1,5000):
    values.append([i])

for value in values:
    if(value[0] % 2 ==0):
        answers.append(0)
    else:
        answers.append(1)

X_train, X_test, y_train, y_test = train_test_split(values, answers, random_state=0)

# print(X_test)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))
