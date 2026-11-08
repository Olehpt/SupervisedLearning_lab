import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from knn import *

def plot_dataset(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="coolwarm",
        marker="o",
        alpha=0.7,
        label="Training data"
    )

    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        marker="x",
        alpha=0.9,
        label="Testing data"
    )

    plt.title("Supervised Learning - Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig("Dataset.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid)
    predictions = predictions.reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="black")

    plt.title("KNN Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("KNN_decision_boundary.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_ROC_curves(models, X, y):
    plt.figure(figsize=(8, 6))

    for name, model in models.items():

        probabilities = model.predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(
            y,
            probabilities
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("ROC_curves.png", dpi=300, bbox_inches="tight")
    plt.close()