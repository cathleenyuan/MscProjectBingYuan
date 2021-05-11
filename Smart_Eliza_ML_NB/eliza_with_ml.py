import logging
import os
import pickle
import random
import re

import numpy as np
from pymongo import MongoClient
from sklearn import preprocessing

import machineLearning.configuration_constants
from machineLearning import multinomialNB_model
from machineLearning import negation_detection
import time
start_time = time.time()
from machineLearning.configuration_constants import ML_INFLUENCED_CHAT_OUTPUT, ML_NEGATIVE_SENTIMENT_LIST,ML_POSTIVE_SENTIMENT_LIST

log = logging.getLogger(__name__)

# SPECIAL_EMOTION_LIST =["regret", "stress", "hopeless"]

class Key:
    def __init__(self, word, weight, decomps):
        self.word = word
        self.weight = weight
        self.decomps = decomps


class Decomp:
    def __init__(self, parts, save, reasmbs):
        self.parts = parts
        self.save = save
        self.reasmbs = reasmbs
        self.next_reasmb_index = 0


class Eliza:
    def __init__(self):
        self.initials = []
        self.finals = []
        self.quits = []
        self.pres = {}
        self.posts = {}
        self.synons = {}
        self.keys = {}
        self.memory = []

    def load(self, path):
        key = None
        decomp = None
        with open(path) as file:
            for line in file:
                if not line.strip():
                    continue
                tag, content = [part.strip() for part in line.split(':')]
                if tag == 'initial':
                    self.initials.append(content)
                elif tag == 'final':
                    self.finals.append(content)
                elif tag == 'quit':
                    self.quits.append(content)
                elif tag == 'pre':
                    parts = content.split(' ')
                    self.pres[parts[0]] = parts[1:]
                elif tag == 'post':
                    parts = content.split(' ')
                    self.posts[parts[0]] = parts[1:]
                elif tag == 'synon':
                    parts = content.split(' ')
                    self.synons[parts[0]] = parts
                elif tag == 'key':
                    parts = content.split(' ')
                    word = parts[0]
                    weight = int(parts[1]) if len(parts) > 1 else 1
                    key = Key(word, weight, [])
                    self.keys[word] = key
                elif tag == 'decomp':
                    parts = content.split(' ')
                    save = False
                    if parts[0] == '$':
                        save = True
                        parts = parts[1:]
                    decomp = Decomp(parts, save, [])
                    key.decomps.append(decomp)
                elif tag == 'reasmb':
                    parts = content.split(' ')
                    decomp.reasmbs.append(parts)

    def _match_decomp_r(self, parts, words, results):
        if not parts and not words:
            return True
        if not parts or (not words and parts != ['*']):
            return False
        if parts[0] == '*':
            for index in range(len(words), -1, -1):
                results.append(words[:index])
                if self._match_decomp_r(parts[1:], words[index:], results):
                    return True
                results.pop()
            return False
        elif parts[0].startswith('@'):
            root = parts[0][1:]
            if not root in self.synons:
                raise ValueError("Unknown synonym root {}".format(root))
            if not words[0].lower() in self.synons[root]:
                return False
            results.append([words[0]])
            return self._match_decomp_r(parts[1:], words[1:], results)
        elif parts[0].lower() != words[0].lower():
            return False
        else:
            return self._match_decomp_r(parts[1:], words[1:], results)

    def _match_decomp(self, parts, words):
        results = []
        if self._match_decomp_r(parts, words, results):
            return results
        return None

    def _next_reasmb(self, decomp):
        index = decomp.next_reasmb_index
        result = decomp.reasmbs[index % len(decomp.reasmbs)]
        decomp.next_reasmb_index = index + 1
        return result

    def _reassemble(self, reasmb, results):
        output = []
        for reword in reasmb:
            if not reword:
                continue
            if reword[0] == '(' and reword[-1] == ')':
                index = int(reword[1:-1])
                if index < 1 or index > len(results):
                    raise ValueError("Invalid result index {}".format(index))
                insert = results[index - 1]
                for punct in [',', '.', ';']:
                    if punct in insert:
                        insert = insert[:insert.index(punct)]
                output.extend(insert)
            else:
                output.append(reword)
        return output

    def _sub(self, words, sub):
        output = []
        for word in words:
            word_lower = word.lower()
            if word_lower in sub:
                output.extend(sub[word_lower])
            else:
                output.append(word)
        return output

    def _match_key(self, words, key):
        output = []
        for decomp in key.decomps:
            results = self._match_decomp(decomp.parts, words)
            if results is None:
                log.debug('Decomp did not match: %s', decomp.parts)
                continue
            log.debug('Decomp matched: %s', decomp.parts)
            log.debug('Decomp results: %s', results)
            results = [self._sub(words, self.posts) for words in results]
            log.debug('Decomp results after posts: %s', results)
            reasmb = self._next_reasmb(decomp)
            log.debug('Using reassembly: %s', reasmb)
            if reasmb[0] == 'goto':
                goto_key = reasmb[1]
                if not goto_key in self.keys:
                    raise ValueError("Invalid goto key {}".format(goto_key))
                log.debug('Goto key: %s', goto_key)
                return self._match_key(words, self.keys[goto_key])
            output = self._reassemble(reasmb, results)

            if decomp.save:
                self.memory.append(output)
                log.debug('Saved to memory: %s', output)
                continue
            return output
        return None

    def respond(self, text):
        if text.lower() in self.quits:
            return None

        text = re.sub(r'\s*\.+\s*', ' . ', text)
        text = re.sub(r'\s*,+\s*', ' , ', text)
        text = re.sub(r'\s*;+\s*', ' ; ', text)
        log.debug('After punctuation cleanup: %s', text)

        words = [w for w in text.split(' ') if w]
        log.debug('Input: %s', words)

        words = self._sub(words, self.pres)
        log.debug('After pre-substitution: %s', words)

        keys = [self.keys[w.lower()] for w in words if w.lower() in self.keys]
        keys = sorted(keys, key=lambda k: -k.weight)
        log.debug('Sorted keys: %s', [(k.word, k.weight) for k in keys])

        output = None

        for key in keys:
            output = self._match_key(words, key)
            if output:
                log.debug('Output from key: %s', output)
                break
        if not output:
            if self.memory:
                index = random.randrange(len(self.memory))
                output = self.memory.pop(index)
                log.debug('Output from memory: %s', output)
            else:
                output = self._next_reasmb(self.keys['xnone'].decomps[0])
                log.debug('Output from xnone: %s', output)

        return " ".join(output)

    def initial(self):
        return random.choice(self.initials)

    def final(self):
        return random.choice(self.finals)

    def run(self , _highest_score_model_file):
        print(self.initial())
        output = []
        while True:
            sent = input('> ')
            # Call the ML model here and detection mood is negative or not then return output acoordingly.
            emotion_detection_result = ml_emotion_detection_detection(sent, _highest_score_model_file)
            print("--- %s seconds get emotion_detection_result ---" % (time.time() - start_time))
            negation_status = negation_detection.negation_handling(sent)
            if (emotion_detection_result in ML_NEGATIVE_SENTIMENT_LIST and negation_status == False) or ((emotion_detection_result in ML_POSTIVE_SENTIMENT_LIST and negation_status == True)):
                print("***========DEBUG=========*** User is negative sentiment")
                output = ML_INFLUENCED_CHAT_OUTPUT
            else:
                print("***========DEBUG=========*** User is positive sentiment")
                output = self.respond(sent)
                print("--- %s seconds return output---" % (time.time() - start_time))
            if output is None:
                break

            print(output)

        print(self.final())

def ml_emotion_detection_detection(input_sentence ,_highest_score_model_file):

    _loaded_model = _highest_score_model_file

    _input_array = np.asarray([input_sentence])

    # Encoding output labels
    _label_encode = preprocessing.LabelEncoder()
    _label_encode.fit_transform(machineLearning.configuration_constants.TRAIN_DATA.sentiment.values)

    _count_vect = multinomialNB_model.extract_count_vector()
    _input_array_count = _count_vect.transform(_input_array)
    _pred_category = _loaded_model.predict(_input_array_count)

    _emotion_detection_result = _label_encode.inverse_transform(_pred_category)
    _emotion_detection_result_text = ' '.join(map(str, _emotion_detection_result))

    print("***======DEBUG======*** : emotion_detection_result............", _emotion_detection_result_text )

    return _emotion_detection_result_text

def get_highest_score_model_file():
    _model_file_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "machineLearning")
    higest_score_model = list(machineLearning.configuration_constants.MONGO_DB[machineLearning.configuration_constants.MONGO_ML_COLLECTION].find({"type": "single_sentence_ml_model"}).sort("score", -1).limit(1))
    if higest_score_model is not None and len(higest_score_model) > 0:
        higest_score_model = higest_score_model[0]
        if higest_score_model.get("model_file") != None:
            _model_file = higest_score_model['model_file']


            if os.path.exists(os.path.join(_model_file_folder_path,_model_file)):
                loaded_model_file = pickle.load(open(os.path.join(_model_file_folder_path,_model_file), 'rb'))
                print("***======DEBUG======***: ML Model exist within the folder, load the ML model.", _model_file)
                return loaded_model_file
            else:
                print("***======DEBUG======***: ML model file does not exist within folder, generate new model file.")
                return pickle.load(open(os.path.join(_model_file_folder_path , multinomialNB_model.main()),'rb'))
    else:
        print("***======DEBUG======***: ML model file does not exist yet, generate new model file.")
        return pickle.load(open(os.path.join(_model_file_folder_path , multinomialNB_model.main()),'rb'))

def main():

    eliza = Eliza()
    eliza.load('doctor.txt')
    print("--- %s seconds start time stamp ---" % (time.time() ))
    loaded_model_file = get_highest_score_model_file()
    print("--- %s seconds loaded_model_file ---" % (time.time() - start_time))
    eliza.run(loaded_model_file)

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    mongo_db = MongoClient(machineLearning.configuration_constants.MONGO_URI)[machineLearning.configuration_constants.DB_NAME]
    logging.basicConfig()
    main()
