import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()


class MongoDB():
    def __init__(self):
        ATLAS_TOKEN = os.environ["ATLAS_TOKEN"]
        ATLAS_USER = os.environ["ATLAS_USER"]
        self.cluster = MongoClient("mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(ATLAS_USER, ATLAS_TOKEN))

    def load_one_record_from_db(self, db_name, collection_name, cols):
        '''

        Args:
            db_name: database name
            collection_name: collection name
            cols: use list to indicate what columns are required, e.g. ['name', 'email']

        Returns: one data record with dataframe format

        '''

        db = self.cluster[db_name]
        data = db[collection_name]

        selection = {'_id': 0}  # select all cols
        if cols:
            for c in cols:
                selection[c] = 1
        df = pd.DataFrame([data.find_one({}, selection)])
        return df

    def load_all_record_from_db(self, db_name, collection_name, cols):
        '''

        Args:
            db_name: database name
            collection_name: collection name
            cols: use list to indicate what columns are required, e.g. ['name', 'email']
        Returns: all data with dataframe format


        '''

        db = self.cluster[db_name]
        data = db[collection_name]

        selection = {'_id': 0}  # select all cols
        if cols:
            for c in cols:
                selection[c] = 1
        df = pd.DataFrame(list(data.find({}, selection)))
        return df

    # more functions to fetch data
