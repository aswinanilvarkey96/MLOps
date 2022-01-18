# Load data
import numpy as np
from sklearn import datasets
iris_X, iris_y = datasets.load_iris(return_X_y=True)

# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
knn.predict(iris_X_test)
print(iris_X_test[0])

data = "5.9,3.2,4.8,1.8"
print(data.split(','))
input_data = list(map(float, data.split(',')))
print("@@",input_data)

# save model
import pickle
with open('models/sklearn.pkl', 'wb') as file:
    pickle.dump(knn, file)
