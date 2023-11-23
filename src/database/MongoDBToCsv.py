import os
from dotenv import load_dotenv
import numpy
from pymongo import MongoClient
import pandas as pd

# Load .env variables
load_dotenv()

# Access variables from .env file
ATLAS_USER = os.getenv("ATLAS_USER")
ATLAS_TOKEN = os.getenv("ATLAS_TOKEN")

client = MongoClient(
    "mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(
        ATLAS_USER, ATLAS_TOKEN))

db = client["scrape"]
col = db["telegram"]

# Query
query = {
    "$and": [
        {"country": "Switzerland"}
    ]
}
# Projection
projection = {
    "_id": 0,          # Exclude _id column by default unless you need it
    "messageText": 1,  # 1 indicates inclusion of the field
    "messageDatetime": 1,
    "country": 1,
    "state": 1,
    "predicted_class": 1
}

# Execute Query with projection
cursor = col.find(query, projection=projection)


# Convert MongoDB data to a pandas dataframe
df = pd.DataFrame(list(cursor))

df = df[df['predicted_class'] != 'Unknown']
df = df[["messageText", "messageDatetime", "state", "predicted_class"]]
df = df[(df['messageText'].str.len() >= 100) & (df['messageText'].str.len() <= 500)]

import tiktoken
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

df['tokens'] = df['messageText'].apply(lambda x: num_tokens_from_string(x, 'cl100k_base'))

print(df.shape)

df.to_csv('src/frontend/data/df_telegram.csv', index=False)