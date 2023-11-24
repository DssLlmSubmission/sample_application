# Sample Application

This application demonstrates a comprehensive workflow of our workflow our paper on Large Language Models in Information Systems: involving data extraction from Telegram, text analysis using BERTopic, data storage and management with MongoDB, and a frontend interface, which uses GPT-4 to summarize the found clusters, for interaction and visualization.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/DssLlmSubmission/sample_application.git
   cd sample_application
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements_whole_project.txt #NOTE requirements.txt is directly used from streamlit hosting and thus does not contain all packages
   ```   

3. **Setup Environment Variables**
   ```bash
   TELEGRAM_API_ID = "YOUR_TELEGRAM_API_ID"
   TELEGRAM_API_HASH = "YOUR_TELEGRAM_API_HASH"
   TELEGRAM_STRING_TOKEN = "YOUR_TELEGRAM_STRING_TOKEN"
   ATLAS_TOKEN = "YOUR_ATLAS_TOKEN"
   ATLAS_USER = "YOUR_ATLAS_USER"
   OPENAI_ORGANIZATION = "YOUR_OPENAI_ORGANIZATION"
   OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
   ```

The given variables need to be placed on PATH or into a .env file. The given API keys can be created at [OpenAI](https://openai.com/blog/openai-api), [MongoDB](https://www.mongodb.com/docs/atlas/getting-started/), and [Telegram](https://core.telegram.org/). 

## Download data, Model Training, and App Deploy

1. **Download Data**
   ```bash
   python src/helper/scraping/telegram_tools/scrapeTelegramChannelMessages.py -i src/telegram_tools/switzerland_groups.txt -o scrape.telegram 
   ```

The following code snippet will download the given Telegram data and place them within a MongoDB.   

2. **Train Model**
   ```bash
   python src/database/MongoDBToCsv.py
   python src/BERTopic/train_BERTopic.py --input_data src/frontend/data/df_telegram.csv --data_type telegram --output_folder PATH_TO_YOUR_MODEL_FOLDER --k_cluster NUMBER_OF_CLUSTERS
   python src/BERTopic/upload_model_HF.py --path PATH_TO_YOUR_MODEL_FOLDER --HF_repo_ID YOUR_HF_REPO_ID
   python src/BERTopic/eval.py --path_or_repoID PATH_TO_YOUR_MODEL_FOLDER_OR_HF_REPO_ID
   python src/BERTopic/apply_BERTopic_mongoDB.py --path_or_repoID PATH_TO_YOUR_MODEL_FOLDER_OR_HF_REPO_ID
   python src/database/MongoDBToCsv.py
   ``` 

The training process initiates with the `MongoDBToCsv.py` script, which converts data from MongoDB into a CSV format. This CSV file is then fed into the `train_BERTopic.py` script to train the BERTopic model. During this training phase, you can specify the output folder for the model and the number of clusters to use. Once the model is trained, it is uploaded to a Hugging Face repository using the `upload_model_HF.py` script, where you need to provide the path to your model folder and your Hugging Face repository ID. The model's performance is evaluated using the `eval.py` script, which works with either a local model path or a Hugging Face repository ID. Next to the Topic Coherence and Topic Diversity, the clusters should be qualitatively evaluated. When we are happy with the given model we push the predictions for each datapoint to MongoDB. The process is ended by another execution of `MongoDBToCsv.py`, where we now also pull the predicted clusters, which we utilize in the next step for the frontend.

3. **Train Model**
   ```bash
   stream run src/frontend/app.py
   ``` 

This Streamlit application is designed to analyze and summarize topics related to the Ukrainian Refugee Crisis on Telegram. It begins by setting up the page layout and configuring the OpenAI API for GPT model interactions. The app allows users to select a topics of interest, and a date range for analysis. It then visualizes the data on a map and a line chart, showing the distribution of topics over time and across different regions. If the user requests a summary, the app samples relevant messages from the Telegram data, generates a prompt, and uses GPT-4 to create a concise summary of the main needs and topics discussed in the selected messages.