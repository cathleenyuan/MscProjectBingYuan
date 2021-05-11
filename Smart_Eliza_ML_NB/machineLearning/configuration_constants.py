import os

import pandas as pd
from pymongo import MongoClient

# Configuration related constants
SAMPLE_FILE_NAME = 'sample_text_emotion.csv'
MONGO_URI = "mongodb://localhost:27777/"
DB_NAME = "eliza_chat_ml"
MONGO_ML_COLLECTION = "ml_model"
MODEL_DIR = 'model\\'
MONGO_DB = MongoClient(MONGO_URI)[DB_NAME]
TRAIN_DATA = pd.read_csv(os.path.join(os.path.dirname(__file__), SAMPLE_FILE_NAME))

# Sentiment related constants :

ML_NEGATIVE_SENTIMENT_LIST = ['empty', 'sadness', 'hate', 'worry', 'boredom', 'anger']
ML_POSTIVE_SENTIMENT_LIST = ['surprise', 'fun', 'relief', 'love', 'enthusiasm', 'happiness']
ML_INFLUENCED_CHAT_OUTPUT = "I am sorry to hear that, how can I help you?"

# sentiment whole list from the sample csv file
# are: {'empty', 'surprise', 'fun', 'relief', 'sadness', 'love', 'enthusiasm', 'hate', 'worry', 'neutral', 'boredom',
#       'anger', 'happiness'}

