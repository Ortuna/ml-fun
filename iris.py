from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris_dataset = load_iris()
knn = KNeighborsClassifier(n_neighbors=1)

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))
