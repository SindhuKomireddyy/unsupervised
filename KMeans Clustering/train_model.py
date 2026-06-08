import pandas as pd
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Create models folder
os.makedirs(
    "models",
    exist_ok=True
)


# Load dataset
df = pd.read_csv(
    "data/iris.csv"
)


# Remove Id column
if "Id" in df.columns:
    df = df.drop(
        "Id",
        axis=1
    )


# Remove Species column
if "Species" in df.columns:
    df = df.drop(
        "Species",
        axis=1
    )


# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df
)


# Train KMeans model
model = KMeans(
    n_clusters=3,
    random_state=42
)


model.fit(
    X_scaled
)


# Save model
with open(
    "models/kmeans_model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


# Save scaler
with open(
    "models/scaler.pkl",
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


print(
    "KMeans Model Trained Successfully"
)