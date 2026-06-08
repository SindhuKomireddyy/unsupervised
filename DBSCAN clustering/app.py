import streamlit as st
import pickle
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors



# Load files

model = pickle.load(
    open(
        "models/dbscan_model.pkl",
        "rb"
    )
)


scaler = pickle.load(
    open(
        "models/scaler.pkl",
        "rb"
    )
)


features = pickle.load(
    open(
        "models/features.pkl",
        "rb"
    )
)



st.title(
    "DBSCAN Wine Clustering App"
)


st.write(
    "Wine Group Prediction"
)



fixed = st.number_input(
    "Fixed Acidity"
)

volatile = st.number_input(
    "Volatile Acidity"
)

sugar = st.number_input(
    "Residual Sugar"
)

alcohol = st.number_input(
    "Alcohol"
)



if st.button(
    "Find Cluster"
):


    acid_ratio = (
        fixed / volatile
    )


    sugar_alcohol = (
        sugar * alcohol
    )


    data = pd.DataFrame(
        [[
            fixed,
            volatile,
            sugar,
            alcohol,
            acid_ratio,
            sugar_alcohol
        ]]
    )


    # add missing columns

    for col in features:

        if col not in data.columns:

            data[col]=0



    data = data[
        features
    ]


    scaled = scaler.transform(
        data
    )


    labels = model.labels_


    st.success(
        f"Wine Cluster : {labels[0]}"
    )