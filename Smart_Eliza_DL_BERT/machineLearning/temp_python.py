from pathlib import Path
import numpy as np

import tensorflow as tf
from bert import tokenization
import bert_classifier

def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

estimator_path = "C:\\CIT_master_paper_important\\output_dir\\1618244107\\"
predict_data_dir = Path("C:\\CIT_master_paper_important\\chat_predict_data")
num_labels = 28

if __name__ == '__main__':

    # Load emotion categories
    with open(bert_classifier.FLAGS.emotion_file, "r") as f:
        all_emotions = f.read().splitlines()
        if bert_classifier.FLAGS.add_neutral:
            all_emotions = all_emotions + ["neutral"]
        idx2emotion = {i: e for i, e in enumerate(all_emotions)}

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:

        # load the exported model to a TensorFlow session
        tf.saved_model.loader.load(sess, ['serve'], estimator_path)

        # Prepare the input for the prediction based on Model input format

        processor = bert_classifier.DataProcessor(num_labels, predict_data_dir)  # set up preprocessor
        predict_fname = "chat_predict.tsv"
        predict_examples = processor.get_examples("chat_prediction", predict_fname)
        tokenizer = tokenization.FullTokenizer(
            vocab_file="C:\\CIT_master_paper_important\\BERT-Base-model_uncased_L-12_H-768_A-12\\vocab.txt ", do_lower_case=False)

        # convert single example to feature. The inputs features represented by Pandas DataFrame.
        feature = bert_classifier.convert_single_example(1, predict_examples[0], bert_classifier.FLAGS.max_seq_length, tokenizer)

        features = {}
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)

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
            if prob >= bert_classifier.FLAGS.pred_cutoff:
                pred_line.extend([emotion, "%.4f" % prob])
            else:
                pred_line.extend(["", ""])

        print(pred_line)

    print("Complete Run..................", prediction)
