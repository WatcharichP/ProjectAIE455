import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load DataFrame
df = pd.read_csv("jobss.csv")

# Select columns of interest
data = df[["Job Title", "Key Skills"]]

# Load pre-trained SentenceTransformer model
model_name = "bert-base-nli-mean-tokens"
model = SentenceTransformer(model_name)

# Define a function to calculate cosine similarity
def calculate_similarity(test):
    # Remove '|' character and encode key skills
    data_copy = data.copy()  # Make a copy to avoid modifying the original DataFrame
    data_copy.loc[:, "Key Skills"] = data_copy["Key Skills"].str.replace('|', '')
    data_copy.loc[:, "Key Skills"] = data_copy["Key Skills"].apply(lambda x: model.encode(x))

    # Encode the test string
    entest = model.encode(test)

    # Convert embeddings to 2D arrays
    entest = np.array(entest).reshape(1, -1)  # Reshape to 2D array
    data_copy.loc[:, "Key Skills"] = data_copy["Key Skills"].apply(lambda x: np.array(x).reshape(1, -1))  # Reshape each embedding to 2D array

    # Calculate cosine similarity
    data_copy["Similarity"] = data_copy["Key Skills"].apply(lambda x: cosine_similarity(entest, x)[0][0])

    # Find the index of the job with the highest similarity
    matching_job_index = data_copy["Similarity"].idxmax()

    # Get the job title of the matching job
    matching_job = data_copy.loc[matching_job_index, "Job Title"]

    return matching_job

# Set page title and favicon
st.set_page_config(page_title="Job Matcher", page_icon=":chart_with_upwards_trend:")

# Set app title and description
st.title("Job Matcher")
st.write("Enter your skills below and find the matching job title.")

# Input field for the user to input the test string
test_input = st.text_input("Enter your skills :")

# Button to calculate the matching job
if st.button("Find Matching Job"):
    if test_input:
        matching_job = calculate_similarity(test_input)
        st.success(f"Matching Job : {matching_job}")
    else:
        st.error("Please enter your skills.")