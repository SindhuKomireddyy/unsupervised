import pandas as pd
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# Create models folder

os.makedirs(
    "models",
    exist_ok=True
)


# Load dataset

df = pd.read_csv(
    "data/winequality.csv"
)


# Fix column names

df.columns = (
    df.columns
    .str.lower()
    .str.replace(" ", "_")
)


print(
    df.columns
)


# -------------------------
# Feature Engineering
# -------------------------


# acidity ratio

df["acid_ratio"] = (
    df["fixed_acidity"] /
    df["volatile_acidity"]
)


# sugar alcohol feature

df["sugar_alcohol"] = (
    df["residual_sugar"] *
    df["alcohol"]
)



# Remove target

if "quality" in df.columns:

    df = df.drop(
        "quality",
        axis=1
    )


# Handle missing values

df = df.fillna(
    df.mean()
)


features = df.columns.tolist()



# Scaling

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    df
)



# DBSCAN Model

model = DBSCAN(
    eps=2,
    min_samples=5
)


clusters = model.fit_predict(
    X_scaled
)


print(
    "Clusters:",
    set(clusters)
)


# Save model

pickle.dump(
    model,
    open(
        "models/dbscan_model.pkl",
        "wb"
    )
)


# Save scaler

pickle.dump(
    scaler,
    open(
        "models/scaler.pkl",
        "wb"
    )
)



# Save feature names

pickle.dump(
    features,
    open(
        "models/features.pkl",
        "wb"
    )
)


print(
    "DBSCAN Model Trained Successfully"
)