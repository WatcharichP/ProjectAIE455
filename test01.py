import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load DataFrame
df = pd.read_csv("jobss.csv")

# Select columns of interest
data = df[["Job Title", "Key Skills"]]

# Test string
test = "Digital Media Planning"

# Load pre-trained SentenceTransformer model
model_name = "bert-base-nli-mean-tokens"
model = SentenceTransformer(model_name)

# Remove '|' character and encode key skills
data.loc[:, "Key Skills"] = data["Key Skills"].str.replace('|', '')
data.loc[:, "Key Skills"] = data["Key Skills"].apply(lambda x: model.encode(x))

# Encode the test string
entest = model.encode(test)

# Convert embeddings to 2D arrays
entest = np.array(entest).reshape(1, -1)  # Reshape to 2D array
data.loc[:, "Key Skills"] = data["Key Skills"].apply(lambda x: np.array(x).reshape(1, -1))  # Reshape each embedding to 2D array

# Calculate cosine similarity
ans = data["Key Skills"].apply(lambda x: cosine_similarity(entest, x)[0][0])

# Find the index of the job with the highest similarity
matching_job_index = ans.idxmax()

# Get the job title of the matching job
matching_job = data.loc[matching_job_index, "Job Title"]

print(matching_job)
