import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from knn import *
from visualization import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#dataset
X, y = make_moons(
    n_samples=1000,
    noise=0.25,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Dataset shape:", X.shape)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

plot_dataset(X_train, y_train, X_test, y_test)

### 1

knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test)

print(f"KNN (in-house) accuracy: {accuracy * 100:.2f}%")

plot_decision_boundary(knn, X_train, y_train)

### 2

knn_other = KNeighborsClassifier(n_neighbors=5)
knn_other.fit(X_train, y_train)

nb = GaussianNB()
nb.fit(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

models = {
    "KNN": knn,
    "KNN out of box" : knn_other,
    "Naive Bayes": nb,
    "Logistic Regression": lr,
    "Decision Tree": dt
}

print("\nmodel comparison:\n")
for name, model in models.items():
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)

    print(f"{name}: {accuracy * 100:.2f}%")

### 3

plot_ROC_curves(models, X_test, y_test)