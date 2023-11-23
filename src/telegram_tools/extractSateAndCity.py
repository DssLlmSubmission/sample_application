from geosky import geo_plug
import pandas as pd
import argparse
import os
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient

def validate_file(f): #function to check if file exists
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def initialize_database(database_name, collection_name):
    '''
    use names of database and collection to fetch specific collection
    Args:
        database_name:
        collection_name:

    Returns:

    '''
    ATLAS_TOKEN = os.environ["ATLAS_TOKEN"]
    ATLAS_USER = os.environ["ATLAS_USER"]
    cluster = MongoClient("mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(ATLAS_USER, ATLAS_TOKEN))
    collection = cluster[database_name][collection_name]
    return collection

def get_country_to_state_dict():
    '''
    prepare country-state mapping
    Returns:
            a mapping with country as key and states as value
            e.g. {'Switzerland':['Zurich', 'Zug', 'Vaud', 'Saint Gallen'...], 'Germany':[]}
    '''

    data_state = geo_plug.all_Country_StateNames()
    data_state = data_state.replace('null', ' ')
    res = eval(data_state)

    mapping_state = {}
    for element in res:
        for k, v in element.items():
            mapping_state[k] = v

    mapping_state["Switzerland"].remove("Basel-City")
    mapping_state["Switzerland"].append("Basel")

    return mapping_state

def get_state_to_city_dict():
    '''
    prepare state-city mapping
    Returns:
            a mapping with states as key and city as value
            e.g. {'Zurich':["Winterthur", "Uster", ...], 'Basel':[]}
    '''

    data_city = geo_plug.all_State_CityNames()
    data_city = data_city.replace('null', ' ')
    res = eval(data_city)

    mapping_city = {}
    for element in res:
        for k, v in element.items():
            mapping_city[k] = v

    mapping_city['North Rhine-Westphalia'].append('Cologne')
    mapping_city['Bavaria'].append('Nuremberg')
    mapping_city['Basel'] = mapping_city.pop('Basel-City')

    return mapping_city

def special_translate_chat(chat):
    '''
    In same chats, they are writen in German or French. This functions will standardize their spelling
    Args:
        chat: original chat (string)

    Returns: chat with standard format

    '''
    return chat.replace("Lousanne", "Lausanne") \
                .replace("BielBienne", "Biel/Bienne")\
                .replace("Geneve", "Geneva") \
                .replace("StGallen", "Saint Gallen")


if __name__ == '__main__':
    """
    This code is to parse state and city for each chat
    We can store output in local file or MongoDB. 
    e.g. When you choose option 1 (store output in local file),  delete code of Option 2 (store in MongoDB).
    
    example usage in command line:
    
    Option 1: write to local file
    python src/helper/scraping/telegram_tools/extractSateAndCity.py -i data/telegram/queries/DACH.txt -o data/telegram/queries/chat_with_country.csv
    
    Option 2: write to database
    python src/helper/scraping/telegram_tools/extractSateAndCity.py -i data/telegram/queries/DACH.txt -o scrape.telegramChatsWithState
    
    """

    # # Option 1: write to local file. Use command below
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input_file_path', help="Specify the input file", type=validate_file, required=True)
    # parser.add_argument('-o', '--output_file_path', help="Specify the output file", required=True)
    # args = parser.parse_args()

    # Option 2: write to database.
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file_path', help="Specify the input file", type=validate_file, required=True)
    parser.add_argument('-o', '--output_database', help="Specify the output database", required=True)
    args = parser.parse_args()
    o_database_name, o_collection_name = args.output_database.split('.')

    mapping_state = get_country_to_state_dict()
    mapping_city = get_state_to_city_dict()

    countries, states, cities, chats = list(), list(), list(), list()
    with open(args.input_file_path, 'r') as file:
        for line in file.readlines():
            if line.startswith("#"):
                country = line.replace('#', '').replace('\n', '')
            else:
                chat = line.replace('\n', '')

                chat_standard = special_translate_chat(chat)

                # parse state and city
                chat_states = mapping_state[country]
                state, city = '', ''
                for s in chat_states:
                    chat_city = mapping_city[s]
                    for c in chat_city:
                        if c.upper() in chat_standard.upper():
                            city = c
                            state = s
                            break

                        if s.upper() in chat_standard.upper():
                            state = s

                chats.append(chat)
                countries.append(country)
                states.append(state)
                cities.append(city)

    # # Option 1: write to local file
    # df = pd.DataFrame(list(zip(countries, states, cities, chats)), columns =['country', 'state', 'city', 'chat'])
    # df.to_csv(args.output_file_path, index=False)

    # Option 2: write to database (Overwrite)
    df = pd.DataFrame(list(zip(countries, states, cities, chats)), columns=['country', 'state', 'city', 'chat'])
    data_list = df.to_dict('records')
    collection = initialize_database(o_database_name, o_collection_name)
    collection.drop()
    collection.insert_many(data_list)





