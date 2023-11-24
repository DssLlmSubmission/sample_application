from pymongo import MongoClient, UpdateOne
import pandas as pd
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from tqdm import tqdm

def fetch_data():
    global col
    load_dotenv('.env')
    ATLAS_USER = os.getenv("ATLAS_USER")
    ATLAS_TOKEN = os.getenv("ATLAS_TOKEN")
    client = MongoClient("mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(ATLAS_USER, ATLAS_TOKEN))
    db = client["scrape"]
    col = db["telegram"]
    query = {"$and": [{"country": "Switzerland"}]}
    cursor = col.find(query)
    df = pd.DataFrame(list(cursor))
    # df = df.iloc[:50000]
    return df, client  # Return the client as well

def predict_topics(df, model_path):
    # Initialize an empty list to store all topics
    all_topics = []

    # Initialize embedding and topic models
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    topic_model = BERTopic.load(model_path, embedding_model=embedding_model)

    # Chunk size
    chunk_size = 100000

    # Loop through DataFrame in chunks
    for i in tqdm(range(0, len(df), chunk_size)):
        chunk = df.iloc[i:i + chunk_size]
        texts = chunk['messageText'].tolist()

        # Perform prediction on the chunk
        topics, _ = topic_model.transform(texts)
        
        all_topics.extend(topics)

    # Add all predictions to the DataFrame
    df['predicted_class_old'] = all_topics

    return df

def cluster_to_clustername(df):

    topic_mapping = {-1: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 0: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Information Requests'}, 1: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 2: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Insurance'}, 3: {'cluster_id': 2, 'cluster_name': 'Pet', 'sub_cluster': 'Pet'}, 4: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Asylum'}, 5: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Ticket Inquiries'}, 6: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Carriers, Transport to and from Ukraine'}, 7: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Seeking'}, 8: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 9: {'cluster_id': 5, 'cluster_name': 'Volunteering', 'sub_cluster': 'Volunteering'}, 10: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Communication'}, 11: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Translation Services'}, 12: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Passport'}, 13: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Dentistry'}, 14: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Job'}, 15: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Currency'}, 16: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Banking'}, 17: {'cluster_id': 8, 'cluster_name': 'Social Services', 'sub_cluster': 'Protocols'}, 18: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Mail'}, 19: {'cluster_id': 9, 'cluster_name': 'Education', 'sub_cluster': 'Education'}, 20: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Clothing'}, 21: {'cluster_id': 8, 'cluster_name': 'Social Services', 'sub_cluster': 'Financial Assistance'}, 22: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 23: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 24: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Carriers, Transport to and from Ukraine'}, 25: {'cluster_id': 9, 'cluster_name': 'Education', 'sub_cluster': 'Education'}, 26: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 27: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Public Transportation'}, 28: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Leasing Regulation'}, 29: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 30: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Open Chat'}, 31: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Communication'}, 32: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 33: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Regulation'}, 34: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Food'}, 35: {'cluster_id': 2, 'cluster_name': 'Pet', 'sub_cluster': 'Pet'}, 36: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Carriers, Transport to and from Ukraine'}, 37: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Vehicle'}, 38: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 39: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 40: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Request'}, 41: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 42: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Status Acquisition'}, 43: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Consulate Services'}, 44: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 45: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Public Transportation'}, 46: {'cluster_id': 5, 'cluster_name': 'Volunteering', 'sub_cluster': 'Volunteering'}, 47: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 48: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Vehicle'}, 49: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Seeking'}, 50: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Immigration Procedure'}, 51: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'War Chat'}, 52: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 53: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Network Provider'}, 54: {'cluster_id': 9, 'cluster_name': 'Education', 'sub_cluster': 'Education'}, 55: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 56: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 57: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Asylum'}, 58: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 59: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Leisure and Fitness'}, 60: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Tax'}, 61: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Expense'}, 62: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Seeking'}, 63: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 64: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 65: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Carriers, Transport to and from Ukraine'}, 66: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 67: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 68: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Public Transportation'}, 69: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Family Reunion'}, 70: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 71: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Request'}, 72: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 73: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Public Transportation'}, 74: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 75: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Vaccinations'}, 76: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Police'}, 77: {'cluster_id': 8, 'cluster_name': 'Social Services', 'sub_cluster': 'Financial Assistance'}, 78: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Regulation'}, 79: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Carriers, Transport to and from Ukraine'}, 80: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 81: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Request'}, 82: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Public Transportation'}, 83: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Parking'}, 84: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 85: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Travel'}, 86: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 87: {'cluster_id': 11, 'cluster_name': 'Legal information', 'sub_cluster': 'Legal information'}, 88: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 89: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 90: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Insurance'}, 91: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Network Provider'}, 92: {'cluster_id': 9, 'cluster_name': 'Education', 'sub_cluster': 'Education'}, 93: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Seeking'}, 94: {'cluster_id': 12, 'cluster_name': 'Religious Information', 'sub_cluster': 'Religious Information'}, 95: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Network Provider'}, 96: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 97: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 98: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 99: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 100: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Banking'}, 101: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 102: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 103: {'cluster_id': 8, 'cluster_name': 'Social Services', 'sub_cluster': 'Library'}, 104: {'cluster_id': 8, 'cluster_name': 'Social Services', 'sub_cluster': 'Library'}, 105: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Tax'}, 106: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Police'}, 107: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Travel'}, 108: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 109: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Network Provider'}, 110: {'cluster_id': 11, 'cluster_name': 'Legal information', 'sub_cluster': 'Legal information'}, 111: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Passport'}, 112: {'cluster_id': 9, 'cluster_name': 'Education', 'sub_cluster': 'Education'}, 113: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Regulation'}, 114: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Immigration Procedure'}, 115: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 116: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 117: {'cluster_id': 9, 'cluster_name': 'Education', 'sub_cluster': 'Education'}, 118: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Job'}, 119: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 120: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 121: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Seeking'}, 122: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Translation Services'}, 123: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Insurance'}, 124: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Seeking'}, 125: {'cluster_id': 11, 'cluster_name': 'Legal information', 'sub_cluster': 'Legal information'}, 126: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 127: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 128: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 129: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Psychotherapy'}, 130: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 131: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 132: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Seeking'}, 133: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Immigration Procedure'}, 134: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 135: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Home Appliances'}, 136: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 137: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 138: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 139: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Tax'}, 140: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Vaccinations'}, 141: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Travel'}, 142: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Request'}, 143: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Shopping'}, 144: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 145: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 146: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 147: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Job'}, 148: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Vehicle'}, 149: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 150: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 151: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 152: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Request'}, 153: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 154: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 155: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 156: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 157: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 158: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Leisure and Fitness'}, 159: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 160: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Communication'}, 161: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Leisure and Fitness'}, 162: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 163: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 164: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Travel'}, 165: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Shopping'}, 166: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 167: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Public Transportation'}, 168: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Communication'}, 169: {'cluster_id': 12, 'cluster_name': 'Religious Information', 'sub_cluster': 'Religious Information'}, 170: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Shopping'}, 171: {'cluster_id': 3, 'cluster_name': 'Transportation', 'sub_cluster': 'Taxi Services'}, 172: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Travel'}, 173: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Travel'}, 174: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 175: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Open Chat'}, 176: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 177: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 178: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Travel'}, 179: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Immigration Procedure'}, 180: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 181: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 182: {'cluster_id': 11, 'cluster_name': 'Legal information', 'sub_cluster': 'Divorce'}, 183: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 184: {'cluster_id': 8, 'cluster_name': 'Social Services', 'sub_cluster': 'Protocols'}, 185: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Shopping'}, 186: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 187: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Immigration Procedure'}, 188: {'cluster_id': 11, 'cluster_name': 'Legal information', 'sub_cluster': 'Marriage'}, 189: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Job'}, 190: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 191: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 192: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Leisure and Fitness'}, 193: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Travel'}, 194: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 195: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 196: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 197: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Leisure and Fitness'}, 198: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Shopping'}, 199: {'cluster_id': 5, 'cluster_name': 'Volunteering', 'sub_cluster': 'Volunteering'}, 200: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Job'}, 201: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Logistics'}, 202: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 203: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Consulate Services'}, 204: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Seeking'}, 205: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Other Item Request'}, 206: {'cluster_id': 4, 'cluster_name': 'Accommodation', 'sub_cluster': 'Leasing Regulation'}, 207: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Other Item Request'}, 208: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Job'}, 209: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 210: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 211: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 212: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 213: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Infant & Toddler Care'}, 214: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 215: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 216: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 217: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Regulation'}, 218: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 219: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Request'}, 220: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 221: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 222: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Hospice Care'}, 223: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 224: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 225: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Request'}, 226: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 227: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Dentistry'}, 228: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 229: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 230: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Customs'}, 231: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 232: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 233: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Customs'}, 234: {'cluster_id': 6, 'cluster_name': 'Integration', 'sub_cluster': 'Customs'}, 235: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Disability'}, 236: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 237: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 238: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 239: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 240: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Regulation'}, 241: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Network Provider'}, 242: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 243: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 244: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 245: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Leisure and Fitness'}, 246: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 247: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Travel'}, 248: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 249: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Leisure and Fitness'}, 250: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Immigration Procedure'}, 251: {'cluster_id': 10, 'cluster_name': 'Social Activity', 'sub_cluster': 'Regulation'}, 252: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 253: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Open Chat'}, 254: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 255: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Medical Request'}, 256: {'cluster_id': 0, 'cluster_name': 'Immigration', 'sub_cluster': 'Immigration Procedure'}, 257: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 258: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 259: {'cluster_id': 8, 'cluster_name': 'Social Services', 'sub_cluster': 'Protocols'}, 260: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 261: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 262: {'cluster_id': -1, 'cluster_name': 'Unknown', 'sub_cluster': 'Unknown'}, 263: {'cluster_id': 1, 'cluster_name': 'Healthcare and Insurance', 'sub_cluster': 'Infant & Toddler Care'}, 264: {'cluster_id': 7, 'cluster_name': 'Living Essentials', 'sub_cluster': 'Shopping'}, 265: {'cluster_id': 5, 'cluster_name': 'Volunteering', 'sub_cluster': 'Volunteering'}}
    df['cluster_id_fit'] = df['predicted_class_old'].map(lambda x: topic_mapping[x]['cluster_id'])
    df['predicted_class'] = df['predicted_class_old'].map(lambda x: topic_mapping[x]['cluster_name'])
    df['sub_cluster'] = df['predicted_class_old'].map(lambda x: topic_mapping[x]['sub_cluster'])
    # df['cluster_name'] = df['predicted_class'].map(lambda x: cluster_mapping[x]['cluster_name'])
    return df

def update_chunk(chunk):
    bulk_operations = [UpdateOne({"_id": row["_id"]}, {"$set": {"predicted_class": row["predicted_class"]}}) for index, row in chunk.iterrows()]
    col.bulk_write(bulk_operations)

def predict_and_update(model_path):
    df, client = fetch_data()  # Get the client as well
    print("data loaded")
    df = predict_topics(df, model_path)

    df = cluster_to_clustername(df)

    print("topics predicted")

    # Splitting dataframe into chunks of 1000
    chunks = [df.iloc[i:i+1000] for i in range(0, len(df), 1000)]

    # Using ThreadPoolExecutor to update chunks in parallel
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(update_chunk, chunks), total=len(chunks)))

    print("Records updated successfully!")
    
    client.close()  # Close the client here after all operations are done

# Example usage
model_path = "DssLlmSubmission/BERTopic_UKR_CH"
predict_and_update(model_path)