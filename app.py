# Importing required packages
import streamlit as st
import openai
import re
import os
import pandas as pd
# from openai.embeddings_utils import get_embedding, cosine_similarity
from azure.identity import DefaultAzureCredential
import numpy as np
import requests
import sys
import json
# pip install streamlit-chat
from streamlit_chat import message
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

st.title("Chat with Mahindra Intelligent bot powered by Azure OpenAI")
st.sidebar.header("Instructions")
st.sidebar.info(
    '''This is a web application that allows you to interact with 
       the Mahindra XUV700 Customer bot.
       Enter a **query** in the **text box** and **press enter** to receive 
       a **response** from the model
       '''
)
# get user query
# pass the user query to CQA prediction url
# call the customqa response
# new code with DefaultAzureCredential
# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


# bot_start = get_text()
user_input = get_text()
default_credential = DefaultAzureCredential()
TEXT_DAVINCI_003 = "text-davinci-003"  # Model
API_KEY = '884b52d664d34d5bbb412e1c4d5076d9'
RESOURCE_ENDPOINT = 'https://adcgpt.openai.azure.com/'
openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"
url = openai.api_base + "/openai/deployments?api-version=2022-12-01"
r = requests.get(url, headers={"api-key": API_KEY})

# index_name = ""
# # Get the service endpoint and API key from the environment
# endpoint = ''
# key = ''
# # Create a client
# credential = AzureKeyCredential(key)
# client = SearchClient(endpoint=endpoint,
#                       index_name=index_name,
#                       credential=credential)
## Function to get azure custom question answer response
def cog_search1(user_input):
    index_name = "manual"
    # Setup the endpoint
    endpoint = 'https://search-gpt.search.windows.net'
    headers = {'Content-Type': 'application/json',
               'api-key': 'dJfhuyHFe3itLQXEvegOeqIvjqH5oW2s6PyDHWENmvAzSeBKYCN6'}
    params = {
        'api-version': '2021-04-30-Preview',
        "search": user_input
    }
    r = requests.get(
        endpoint + "/indexes/" + index_name + "/docs?&$top=2&queryLanguage=en-US&queryType=semantic&captions=extractive&answers=extractive%7Ccount-3&semanticConfiguration=mahindrapoc",
        headers=headers, params=params)
    # r = requests.get("https://search-gpt.search.windows.net/indexes/manual/docs?api-version=2021-04-30-Preview&search=adrenox%20xuv700&queryLanguage=en-US&queryType=semantic&captions=extractive&answers=extractive%7Ccount-3&semanticConfiguration=mahindrapoc")

    y = json.loads(r.text)
    for doc in y['@search.answers']:
        highlights = doc['text']
    for doc in y['value']:
        content = doc['content'][:9000]
    srch_response = content  # highlights+content
    return (srch_response)  # highlights to be added


def generate_prompt(srch_response, user_input):
    prompt = f"""Respond to a user as a chatbot that is answering questions for a customer of a Indian automobile focused on automotive manufacturing based out of India,  on their internal documents, policies and FAQ.Use the following information in the response if possible. Please reply to the question using only the information present in the text above,If you can't find it, reply with a lesser confidence : {srch_response} \nUser Question: {user_input} \nBot Answer:"""
    print(prompt)
    return prompt


def generate_prompt_turns(srch_response, user_input, past_user_inputs, generated_responses):
    past_user_inputs_turn_1 = ''
    if (len(past_user_inputs) == 0):
        past_user_inputs_turn_1 = user_input
    else:
        past_user_inputs_turn_1 = past_user_inputs[-1]
    #    print("past_user_inputs_turn_1 ", past_user_inputs_turn_1)
    generated_responses_turn_1 = ''
    if (len(generated_responses) == 0):
        generated_responses_turn_1 = srch_response
    else:
        generated_responses_turn_1 = generated_responses[-1]
    prompt = f"""Respond to a user as a chatbot that is answering questions for a customer of a Indian automobile conglomerate focused on automotive manufacturing based out of India,  on their internal documents, policies and FAQ.Use the following information in the response if possible. Please reply to the question using only the information present in the text above,If you can't find it, reply with a lesser confidence : {srch_response} \nUser Question: {past_user_inputs_turn_1} \nBot Answer: {generated_responses_turn_1} \nUser Question: {user_input} \nBot Answer:"""
    print(prompt)
    return prompt


def generate_response(prompt):
    completions = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0.1,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.2,
        presence_penalty=0.2,
        best_of=1
    )
    message = completions.choices[0].text
    return message


# Creating the chatbot interface
if user_input:
    #    srch_highlights, content=cog_search1(user_input)
    srch_response = cog_search1(user_input)
    print("search response ", srch_response)
    past_user_inputs = st.session_state.past
    generated_responses = st.session_state.generated
    prompt = generate_prompt(srch_response, user_input)
    #    prompt = generate_prompt_turns(srch_response,user_input,past_user_inputs,generated_responses)
    print("prompt ", prompt)
    output = generate_response(prompt)
    print("output ", output)
    # store the output
    # user_context = user_input
    # bot_context = output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
