import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Function for data cleaning and preprocessing
def preprocess_data(df):
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Drop rows with null or empty values in "Key Skills" column
    df.dropna(subset=["Key Skills"], inplace=True)

    # Text cleaning: Remove special characters and convert text to lowercase
    df["Key Skills"] = df["Key Skills"].str.replace("[^\w\s#.+]", "").str.lower()

    # Remove '|' character from the "Key Skills" column
    df["Key Skills"] = df["Key Skills"].str.replace('|', '')

    return df

# Load DataFrame and preprocess data
df = pd.read_csv("jobss.csv")
df = preprocess_data(df)

# Select columns of interest
data = df[["Job Title", "Key Skills"]]

# Load pre-trained SentenceTransformer model
model_name = "bert-base-nli-mean-tokens"
model = SentenceTransformer(model_name)

# Encode the key skills
data.loc[:, "Key Skills"] = data["Key Skills"].apply(lambda x: model.encode(x))

# Define a function to preprocess text and calculate cosine similarity
def calculate_similarity(test):
    # Clean and preprocess the test string
    test_cleaned = re.sub(r'[^\w\s]', '', test).lower()

    # Encode the test string
    entest = model.encode(test_cleaned)

    # Calculate cosine similarity
    data["Similarity"] = data["Key Skills"].apply(lambda x: cosine_similarity([entest], [x])[0][0])

    # Find the index of the job with the highest similarity
    matching_job_index = data["Similarity"].idxmax()

    # Get the job title of the matching job
    matching_job = data.loc[matching_job_index, "Job Title"]

    return matching_job

# Set page title and favicon
st.set_page_config(page_title="Job Matcher", page_icon=":chart_with_upwards_trend:")

# Set app title and description
st.title("Job Matcher")
st.write("Enter your skills below and find the matching job title.")

# Input field for the user to input the test string
test_input = st.text_input("Enter your skills :")

# # Button to calculate the matching job
# if st.button("Find Matching Job"):
#     if test_input:
#         matching_job = calculate_similarity(test_input)
#         st.success(f"Matching Job : {matching_job}")
#     else:
#         st.error("Please enter your skills.")

if st.button("Find Matching Job"):
    if test_input:
        matching_job = calculate_similarity(test_input)
        matching_score = data.loc[data["Job Title"] == matching_job, "Similarity"].iloc[0]
        st.success(f"Matching Job: {matching_job}")
        st.info(f"Matching Score: {matching_score}")
    else:
        st.error("Please enter your skills.")
