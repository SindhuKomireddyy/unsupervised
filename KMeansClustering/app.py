import streamlit as st
import pickle
import numpy as np


# Load model

with open(
    "models/kmeans_model.pkl",
    "rb"
) as f:

    model = pickle.load(f)



# Load scaler

with open(
    "models/scaler.pkl",
    "rb"
) as f:

    scaler = pickle.load(f)



st.title(
    "KMeans Clustering App"
)


st.write(
    "Iris Flower Cluster Prediction"
)


# User Inputs

sepal_length = st.number_input(
    "Sepal Length (cm)"
)


sepal_width = st.number_input(
    "Sepal Width (cm)"
)


petal_length = st.number_input(
    "Petal Length (cm)"
)


petal_width = st.number_input(
    "Petal Width (cm)"
)



if st.button(
    "Predict Cluster"
):

    data = np.array(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]
    )


    scaled_data = scaler.transform(
        data
    )


    prediction = model.predict(
        scaled_data
    )


    st.success(
        f"Flower belongs to Cluster: {prediction[0]}"
    )


    if prediction[0] == 0:

        st.info(
            "Cluster 0"
        )

    elif prediction[0] == 1:

        st.info(
            "Cluster 1"
        )

    else:

        st.info(
            "Cluster 2"
        )