import os
from dotenv import load_dotenv
import numpy
from pymongo import MongoClient
import pandas as pd

# Load environment variables from a .env file
load_dotenv()

# Access MongoDB credentials from environment variables
ATLAS_USER = os.getenv("ATLAS_USER")
ATLAS_TOKEN = os.getenv("ATLAS_TOKEN")

# Establish a connection to the MongoDB cluster using credentials
client = MongoClient(
    f"mongodb+srv://{ATLAS_USER}:{ATLAS_TOKEN}@cluster0.fcobsyq.mongodb.net/"
)

# Select the database and collection
db = client["scrape"]
col = db["telegram"]

# Define a query to filter documents
query = {
    "$and": [
        {"country": "Switzerland"}
    ]
}

# Define a projection to specify which fields to include in the result
projection = {
    "_id": 0,  # Exclude the _id field
    "messageText": 1,  # Include messageText field
    "messageDatetime": 1,  # Include messageDatetime field
    "country": 1,  # Include country field
    "state": 1,  # Include state field
    "predicted_class": 1  # Include predicted_class field
}

# Execute the query with the specified projection
cursor = col.find(query, projection=projection)

# Convert the query results to a pandas DataFrame
df = pd.DataFrame(list(cursor))

# Filter out rows where 'predicted_class' is 'Unknown'
df = df[df['predicted_class'] != 'Unknown']

# Select specific columns for further analysis
df = df[["messageText", "messageDatetime", "state", "predicted_class"]]

# Filter messages based on their length
df = df[(df['messageText'].str.len() >= 100) & (df['messageText'].str.len() <= 500)]

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculate the number of tokens in a text string.

    Args:
    string (str): The text string to be tokenized.
    encoding_name (str): The name of the encoding to use for tokenization.

    Returns:
    int: The number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Apply the tokenization function to each message and store the result in a new column
df['tokens'] = df['messageText'].apply(lambda x: num_tokens_from_string(x, 'cl100k_base'))

# Print the shape of the DataFrame
print(df.shape)

# Save the DataFrame to a CSV file
df.to_csv('src/frontend/data/df_telegram.csv', index=False)
