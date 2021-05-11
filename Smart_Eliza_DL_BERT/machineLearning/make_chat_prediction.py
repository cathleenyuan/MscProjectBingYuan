import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import csv
from .bert import tokenization
from .bert_classifier import FLAGS,DataProcessor,convert_single_example
from .goemotions_configuration_constant import ESTIMATOR_PATH, PREDICT_DATA_DIR,VOL_PATH

num_labels = 28

tf.get_logger().setLevel(logging.ERROR)

class emotion_prediction_class:

    def __init__(self, chat_input_text):
        self.chat_input_text = chat_input_text
        chat_file = Path(PREDICT_DATA_DIR, "chat_predict.tsv")
        sent_list = chat_input_text.split('\n')
        with open(chat_file, 'w', encoding='utf8', newline='') as csvfile:
            tsv_writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n')
            tsv_writer.writerow(sent_list)
            csvfile.close()

    def create_int_feature(self, values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    def make_emotion_prediction(self):
        # Load emotion categories
        with open(FLAGS.emotion_file, "r") as f:
            all_emotions = f.read().splitlines()
            if FLAGS.add_neutral:
                all_emotions = all_emotions + ["neutral"]
            idx2emotion = {i: e for i, e in enumerate(all_emotions)}

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:

            # load the exported model to a TensorFlow session
            tf.saved_model.loader.load(sess, ['serve'], ESTIMATOR_PATH)

            # Prepare the input for the prediction based on Model input format

            processor = DataProcessor(num_labels, PREDICT_DATA_DIR)  # set up preprocessor
            predict_fname = "chat_predict.tsv"

            predict_examples = processor.get_examples("chat_prediction", predict_fname)
            tokenizer = tokenization.FullTokenizer(
                vocab_file=VOL_PATH , do_lower_case=False)

            # convert single example to feature. The inputs features represented by Pandas DataFrame.
            feature = convert_single_example(1, predict_examples[0], FLAGS.max_seq_length, tokenizer)

            features = {}
            features["input_ids"] = self.create_int_feature(feature.input_ids)
            features["input_mask"] = self.create_int_feature(feature.input_mask)
            features["segment_ids"] = self.create_int_feature(feature.segment_ids)
            features["label_ids"] = self.create_int_feature(feature.label_ids)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))

            # https://stackoverflow.com/questions/45900653/tensorflow-how-to-predict-from-a-savedmodel
            prediction = sess.run(
                'loss/Softmax:0',
                feed_dict={"input_example_tensor:0": [tf_example.SerializeToString()]})

            sorted_idx = np.argsort(-prediction[0])
            top_3_emotion = [idx2emotion[idx] for idx in sorted_idx[:3]]
            top_3_prob = [prediction[0][idx] for idx in sorted_idx[:3]]
            pred_line = []
            for emotion, prob in zip(top_3_emotion, top_3_prob):
                if prob >= FLAGS.pred_cutoff:
                    pred_line.extend([emotion, "%.4f" % prob])
                else:
                    pred_line.extend(["", ""])
            # print("***========DEBUG=========*** First Three emotions are", pred_line)
            return pred_line[0]




