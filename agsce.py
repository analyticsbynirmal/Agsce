# Simple ML Program: Iris Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Choose a model (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=3)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Test the model
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# 6. Make a prediction
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example flower measurements
prediction = model.predict(sample)
print("Predicted species:", iris.target_names[prediction][0])
