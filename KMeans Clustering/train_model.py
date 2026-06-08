import pickle
import os

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# create model folder
os.makedirs(
    "models",
    exist_ok=True
)


# Load Iris dataset

iris = load_iris()

X = iris.data


# Feature scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# KMeans model

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


model.fit(
    X_scaled
)


# Save model

with open(
    "models/kmeans_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# Save scaler

with open(
    "models/scaler.pkl",
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


print(
    "KMeans model saved successfully"
)