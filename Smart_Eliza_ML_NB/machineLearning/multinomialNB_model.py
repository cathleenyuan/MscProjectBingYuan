import os
import pickle
import uuid
import re
import datetime
import time
import pandas as pd

from calendar import timegm
from nltk.corpus import stopwords
from textblob import Word

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from .configuration_constants import MODEL_DIR, MONGO_DB, MONGO_ML_COLLECTION, TRAIN_DATA


#Correcting Letter Repetitions

def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def prepare_data():

    train_data = TRAIN_DATA.drop('author', axis=1)

    # Making all letters lowercase
    train_data['content'] = train_data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    # Removing Punctuation, Symbols
    train_data['content'] = train_data['content'].str.replace('[^\w\s]', ' ')

    # nltk.download('stopwords')
    # Removing Stop Words using NLTK
    stop = stopwords.words('english')
    train_data['content'] = train_data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # Lemmatisation
    train_data['content'] = train_data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    train_data['content'] = train_data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

    # Code to find the top 10,000 rarest words appearing in the train_data
    freq = pd.Series(' '.join(train_data['content']).split()).value_counts()[-10000:]

    # Removing all those rarely appearing words from the train_data
    freq = list(freq.index)
    train_data['content'] = train_data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    return train_data['content']

def main():

    # Encoding output labels
    label_encode = preprocessing.LabelEncoder()
    y = label_encode.fit_transform(TRAIN_DATA.sentiment.values)

    # Splitting into training and testing data in 80:20 ratio
    X_train, X_val, y_train, y_val = train_test_split(TRAIN_DATA.content.values, y, stratify=y,
                                                      random_state=42, test_size=0.2, shuffle=True)

    count_vect = extract_count_vector()
    X_train_count = count_vect.transform(X_train)
    X_val_count = count_vect.transform(X_val)

    # ## Building models using count vectors feature
    # # Model  Multinomial Naive Bayes Classifier
    nb_model = MultinomialNB()
    nb_model.fit(X_train_count, y_train)
    y_pred = nb_model.predict(X_val_count)
    _accuracy_score = accuracy_score(y_pred, y_val)
    print('***======DEBUG======***: Call the ML model: naive bayes count vectors accuracy %s' % accuracy_score(y_pred, y_val))

    # save the model to disk
    # if the file path does not exist,create that file path.
    if os.path.isdir(os.path.join(os.path.dirname(__file__),MODEL_DIR)) is False:
        os.mkdir(os.path.join(os.path.dirname(__file__), MODEL_DIR))

    model_filename = MODEL_DIR + 'model_' + str(uuid.uuid4().hex) + str(
        datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S"))

    pickle.dump(nb_model, open(os.path.join(os.path.dirname(__file__) , model_filename), 'wb'))
    print("***======DEBUG======*** :Saved the file into local file system with model_file_path is: ",model_filename)

    # Update Mongo for the current document and save the Model_file.
    _status = MONGO_DB[MONGO_ML_COLLECTION].insert_one({"type": "single_sentence_ml_model",
                                                        "model_algorithm": "MultinomialNB",
                                                        "model_file": model_filename,
                                                        "score": _accuracy_score,
                                                        "model_created_time": datetime.datetime.now().strftime("%d_%m_%Y %H_%M_%S"),
                                                        "document_created_time": timegm(time.strptime(str(datetime.datetime.now()), "%Y-%m-%d %H:%M:%S.%f"))*1000
                                                        })
    print("***======DEBUG======*** : Save model data into MongoDB.",model_filename)
    return model_filename


def extract_count_vector():
    # # Extracting Count Vectors Parameters
    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(prepare_data())

    return count_vect


if __name__ == '__main__':

    # prepare_data()
    # sentiment_list = set(TRAIN_DATA.sentiment.values)
    main()

