import sys

sys.path.append("/home/hacene/Documents/review_classifier_BERT/")
import streamlit as st

from review_cl import inference as inf
from utils import predict_fn
import json
import pandas as pd
import matplotlib.pyplot as plt

# import review_cl.inference

st.title("Review Classification Using BERT Model")
classes = ["Negative", "Neutral", "Positive"]

indices_to_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}


@st.cache()
def load_artifacts():
    artifacts = json.loads(open("logs/config.json", "r").read())
    artifacts["model"] = inf.model_fn(model_dir="logs/")
    return artifacts


artifacts = load_artifacts()
text_input = st.text_input("Input review to classify")

if text_input:
    predictions = predict_fn(review_body=text_input, model=artifacts["model"])
    predicted_class = indices_to_labels[predictions["predicted_label"]]
    st.markdown(
        "This review is classified as **{}** with a probability of **{:.2f}** ".format(
            predicted_class, predictions["probability"]
        )
    )
    fig, ax = plt.subplots()
    ax.bar(classes, predictions["probabilities"], color=["red", "grey", "green"])
    ax.set_xlabel("Classes")
    ax.set_ylabel("Probabilities")
    plt.grid()
    st.pyplot(fig)
