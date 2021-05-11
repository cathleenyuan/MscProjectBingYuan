
# Constant for bert_classifier.py file
from pathlib import Path

EMOTION_FILE =  "C:\\0.Data_Science_and_Data_analystic\\0.CIT\AI_Master\\4.FourthTerm\Code\Eliza\eliza_ml_goemotions_bert\machineLearning\data\emotions.txt"
DATA_DIR =  "C:\\0.Data_Science_and_Data_analystic\\0.CIT\AI_Master\\4.FourthTerm\Code\Eliza\eliza_ml_goemotions_bert\machineLearning\data"
SENTIMENT_FILE = "C:\\0.Data_Science_and_Data_analystic\\0.CIT\AI_Master\\4.FourthTerm\\Code\\Eliza\eliza_ml_goemotions_bert\\\machineLearning\data\\sentiment_dict.json"

# Constant for use_bert_model_make_chat_prediction file
ESTIMATOR_PATH = "C:\\CIT_master_paper_important\\output_dir\\1618244107\\"
PREDICT_DATA_DIR = Path("C:\\CIT_master_paper_important\\chat_predict_data")
VOL_PATH = "C:\\CIT_master_paper_important\\BERT-Base-model_uncased_L-12_H-768_A-12\\vocab.txt"

# Sentiment related constants :

ML_NEGATIVE_SENTIMENT_LIST = ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"]
ML_POSTIVE_SENTIMENT_LIST = ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"]
ML_AMBIGUOUS_SENTIMENT_LIST = ["realization", "surprise", "curiosity", "confusion"]
ML_INFLUENCED_CHAT_OUTPUT = "I am sorry to hear that,how can I help you?"
