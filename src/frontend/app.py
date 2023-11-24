import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import leafmap.foliumap as leafmap
import openai
import tiktoken
import os 
import json

# Config must be first line in script
st.set_page_config(layout="wide")

# Set OpenAI API key and variables for usage   
openai.organization = os.environ["OPENAI_ORGANIZATION"]
openai.api_key = os.getenv("OPENAI_API_KEY") 

max_input_tokens=3900
max_tokens_output=500
encoding = "cl100k_base"

# Initialize session state for language selection
if 'language' not in st.session_state:
    st.session_state.language = "🇩🇪 Deutsch"

# Load translation data from a JSON file
with open("src/frontend/data/translate_app.json", "r") as f:
    translator = json.load(f)



# calculate number of tokens in a text string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculate the number of tokens in a given text string based on a specified encoding.

    Parameters:
    string (str): The text string to be tokenized.
    encoding_name (str): The name of the encoding to use for tokenization.

    Returns:
    int: The number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# run gpt
def run_gpt(prompt: str, max_tokens_output: int, timeout: int = 10) -> str:
    """
    Generate a response from the GPT model based on a given prompt.

    Parameters:
    prompt (str): The input prompt to the GPT model.
    max_tokens_output (int): The maximum number of tokens in the model's response.
    timeout (int): The time limit for the model's response.

    Returns:
    str: The content of the response from the GPT model.
    """
    completion = openai.ChatCompletion.create(
      model = 'gpt-4',
      messages = [
        {'role': 'user', 'content': prompt}
      ],
      max_tokens = max_tokens_output,
      n = 1,
      stop = None,
      temperature=0,
      timeout=timeout
    )
    return completion['choices'][0]['message']['content']

def start_prompt_creator(cluster: list) -> tuple:
    """
    Create a start prompt for the GPT model based on the cluster.

    Parameters:
    cluster (list): A list of clusters/topics to include in the prompt.

    Returns:
    tuple: A tuple containing the start prompt and the cluster string.
    """
    if len(cluster) > 1:
        cluster = ", ".join(cluster)
    else:
        cluster = cluster[0]

    if st.session_state.language == "🇬🇧 English":
        start_prompt = f"Looking at these messages about {cluster}, what are the up to 5 top needs of refugees regarding {cluster}? Response in English."
    elif st.session_state.language == "🇩🇪 Deutsch":
        start_prompt = f"Looking at these messages about {cluster}, what are the up to 5 top needs of refugees regarding {cluster}? Response in German."

    return start_prompt, cluster

# sample from df
def sample_df_gpt_analysis(df: pd.DataFrame, start_prompt: str, max_input_tokens: int) -> str:
    """
    Sample text from a DataFrame for GPT analysis, ensuring the token limit is not exceeded.

    Parameters:
    df (pd.DataFrame): The DataFrame to sample from.
    start_prompt (str): The initial prompt for the GPT model.
    max_input_tokens (int): The maximum number of input tokens allowed.

    Returns:
    str: A string containing the sampled text.
    """
    current_input_tokens = num_tokens_from_string(start_prompt, encoding_name=encoding)
    text_list = []
    text_list.append(start_prompt)
    while max_input_tokens > current_input_tokens:
        df_sample = df.sample(n=1, replace=False)
        df = df.drop(df_sample.index)
        current_input_tokens += df_sample["tokens"].values[0]
        if current_input_tokens > max_input_tokens:
            break
        text_list.append(df_sample["messageText"].values[0])
    
    text = '\n'.join(text_list)
    return text

# write output to streamlit
def write_output(text: str, cluster: str) -> None:
    """
    Write the output text to the Streamlit interface.

    Parameters:
    text (str): The text to be displayed.
    cluster (str): The cluster/topic of the summary.
    """
    st.header(translator[st.session_state.language]["Your Summary 😊"])
    st.write(text)

# load geopandas data
gdf = gpd.read_file("src/frontend/data/germany_switzerland.geojson")

#functions to load data
@st.cache()
def load_telegram_data() -> pd.DataFrame:
    """
    Load Telegram data from a CSV file, with caching to improve performance.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded Telegram data.
    """
    df = pd.read_csv("src/frontend/data/df_telegram.csv")
    #print(df.head(1))
    df['date'] = pd.to_datetime(df['messageDatetime'], utc=True).dt.date
    return df

country_select = "Switzerland"

# manipulate data
def create_df_value_counts(df):
    """
    This function takes a DataFrame and returns a new DataFrame with counts of unique values.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    df_value_counts (DataFrame): The output DataFrame with counts of unique values.
    """
    # Get the count of each unique date in the DataFrame
    messages_per_week_dict = dict(df.value_counts("date"))

    # Get the count of each unique combination of predicted_class and date
    df_value_counts = df.value_counts(["predicted_class", "date"]).reset_index()

    # Rename the columns of the new DataFrame
    df_value_counts.columns = ["predicted_class", "date", "occurence_count"]

    return df_value_counts

def modify_df_for_table(df_mod, country_select, state_select, cluster_select, date_slider, metric_select=None):
    """
    This function modifies a DataFrame based on the selected country, state, cluster, and date range.

    Parameters:
    df_mod (DataFrame): The input DataFrame.
    country_select (str): The selected country.
    state_select (str): The selected state.
    cluster_select (str): The selected cluster.
    date_slider (list): The selected date range.
    metric_select (str, optional): The selected metric. Defaults to None.

    Returns:
    df_mod (DataFrame): The modified DataFrame.
    """
    # Filter the DataFrame based on the selected state
    if state_select not in [translator[st.session_state.language]["all states analysed"]]:
        df_mod = df_mod[df_mod.state==state_select]

    # Filter the DataFrame based on the selected cluster
    if not translator[st.session_state.language]["all found topics"] in cluster_select:
        df_mod = df_mod[df_mod.predicted_class.isin(cluster_select)]

    # Filter the DataFrame based on the selected date range
    df_mod = df_mod[df_mod.date.between(date_slider[0], date_slider[1])]

    return df_mod



# load data
df_telegram = load_telegram_data()

with st.sidebar:
    # Select language
    language_select = st.selectbox(
        'Sprache/Language',
        options=["🇩🇪 Deutsch", "🇬🇧 English"],
        index=["🇩🇪 Deutsch", "🇬🇧 English"].index(st.session_state.language)
    )
    # Update session state if language selection changes
    if st.session_state.language != language_select:
        st.session_state.language = language_select
    
    # Select topics of interest within the telegram data
    cluster_select_telegram = st.multiselect(
        translator[st.session_state.language]['Choose the topics of interest within the telegram data'],
        [translator[st.session_state.language]["all found topics"]] + df_telegram.predicted_class.unique().tolist(),
        [translator[st.session_state.language]["all found topics"]]
        )
    st.write("\n")
    st.write("\n")

    # Button to prepare a summary
    calculate_summary = st.button(translator[st.session_state.language]["prepare summary"])

# Set the title of the page
st.title(translator[st.session_state.language]['Identification of the most relevant topics in the context of the Ukrainian Refugee Crisis on Telegram'])

# create text columns for country, state and time selection
text_col1, text_col2 = st.columns(2)
with text_col1:
    # Select a state of interest
    states = [translator[st.session_state.language]["all states analysed"]] + gdf[gdf["country"]=="Switzerland"].state.unique().tolist()
    state_select = st.selectbox(
        translator[st.session_state.language]['Choose a state of interest'],
        states,
        )
with text_col2:
    # Select a date range of interest
    date_slider = st.slider(translator[st.session_state.language]['Choose date range of interest'],
        min_value=df_telegram.date.min(), 
        value=(df_telegram.date.min(), df_telegram.date.max()), 
        max_value=df_telegram.date.max()
        )

# Modify the dataframe based on the selected state and date range
df_telegram_mod = modify_df_for_table(df_mod=df_telegram, country_select="Switzerland", state_select=state_select, cluster_select=cluster_select_telegram, date_slider=date_slider)
df_value_counts_telegram = create_df_value_counts(df=df_telegram_mod)

# Create columns for visualizations
visual_col1, visual_col2= st.columns(2)
with visual_col1:
    # Create a map visualization based on the selected state
    if state_select==translator[st.session_state.language]["all states analysed"]:
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7)
        m.add_gdf(gdf[gdf["country"]=="Switzerland"], layer_name="Countries Choosen", fill_colors=["red"])
        m.to_streamlit()
    
    if state_select!=translator[st.session_state.language]["all states analysed"]:
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7)
        m.add_gdf(gdf[gdf["state"]!=state_select], layer_name="Countries", fill_colors=["blue"])
        m.add_gdf(gdf[gdf["state"]==state_select], layer_name="Countries Choosen", fill_colors=["red"])
        m.to_streamlit()

with visual_col2:
    # Create a line chart showing the occurrence of topics over time
    if country_select==translator[st.session_state.language]["Germany"] or country_select==translator[st.session_state.language]["Switzerland"] or country_select==translator[st.session_state.language]["all countries analysed"]:
        title_diagram_telegram = translator[st.session_state.language]["Topics over time on Telegram within"] + " " + country_select
        fig = px.line(df_value_counts_telegram.sort_values(['date']), x="date", y="occurence_count", color='predicted_class', title=title_diagram_telegram)
    else:
        title_diagram_telegram = translator[st.session_state.language]["Topics over time on News within"] + " " + state_select
        fig = px.line(df_value_counts_telegram.sort_values(['date']), x="date", y="occurence_count", color='predicted_class', title=title_diagram_telegram)
    fig.update_xaxes(title_text=translator[st.session_state.language]["Date"])
    fig.update_yaxes(title_text=translator[st.session_state.language]["Count"])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<p style='margin-top: 150px;'</p>", unsafe_allow_html=True)

# If the 'prepare summary' button is clicked, generate a summary of the data
if calculate_summary:
    df_mod = df_telegram_mod
    cluster = cluster_select_telegram

    # Display a loading message while the summary is being generated
    dummy_text_summary = st.header(translator[st.session_state.language]["Creating your summary ⏳😊"])
    start_prompt, cluster_str = start_prompt_creator(cluster=cluster)
    prompt = sample_df_gpt_analysis(df=df_mod, start_prompt=start_prompt, max_input_tokens=max_input_tokens-max_tokens_output)
    try:
        # Generate the summary using GPT
        text = run_gpt(prompt, max_tokens_output, timeout=10)
    except openai.OpenAIError as e:
        # If the request times out, display an error message
        text = translator[st.session_state.language]["Sorry, request timed out. Please try again."]
    # Remove the loading message
    dummy_text_summary.empty()
    # Display the generated summary
    write_output(text, cluster_str)
