import numpy as np

class KNNClassifier:

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_one(self, point):
        distances = np.sqrt(np.sum((self.X_train - point) ** 2, axis=1))

        nearest_points = np.argsort(distances)[:self.k]
        nearest_classes = self.y_train[nearest_points]

        c0_count = 0
        c1_count = 0
        for c in nearest_classes:
            if c == 0:
                c0_count += 1
            else:
                c1_count += 1

        if c0_count > c1_count:
            return 0
        else:
            return 1

    def predict(self, X):

        return np.array([
            self.predict_one(point)
            for point in X
        ])

    def predict_proba(self, X):
        probabilities = []

        for point in X:
            distances = np.sqrt(
                np.sum((self.X_train - point) ** 2, axis=1)
            )

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_classes = self.y_train[nearest_indices]

            class_1_count = 0

            for c in nearest_classes:
                if c == 1:
                    class_1_count += 1

            probability = class_1_count / self.k
            probabilities.append([1 - probability, probability])

        return np.array(probabilities)