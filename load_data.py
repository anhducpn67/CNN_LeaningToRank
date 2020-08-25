import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import data_helpers

# Load data
train_data_path = "C:/Users/AnhDuc/Desktop/Study/LabAIDANTE/cnn_learning_to_rank/data/TRAIN-ALL.xml"
test_data_path = "C:/Users/AnhDuc/Desktop/Study/LabAIDANTE/cnn_learning_to_rank/data/TEST.xml"
dev_data_path = "C:/Users/AnhDuc/Desktop/Study/LabAIDANTE/cnn_learning_to_rank/data/DEV.xml"

train_question, train_answer, train_label, train_info = data_helpers.create_data(train_data_path)
dev_question, dev_answer, dev_label, dev_info = data_helpers.create_data(dev_data_path)
test_question, test_answer, test_label, test_info = data_helpers.create_data(test_data_path)

# Create x_feat
train_x_feat = data_helpers.get_x_feat(train_question, train_answer)
dev_x_feat = data_helpers.get_x_feat(dev_question, dev_answer)
test_x_feat = data_helpers.get_x_feat(test_question, test_answer)

# Build vocab
# vocab = set()
# vocab.update(data_helpers.get_vocab(train_question, train_answer))
# vocab.update(data_helpers.get_vocab(dev_question, dev_answer))
# vocab.update(data_helpers.get_vocab(test_question, test_answer))
# vocab = list(vocab)
with open('vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)
vocab_size = len(vocab) + 1

# Encode sentences
encoder = tfds.features.text.TokenTextEncoder(vocab)
train_question = [encoder.encode(sent) for sent in train_question]
train_answer = [encoder.encode(sent) for sent in train_answer]
dev_question = [encoder.encode(sent) for sent in dev_question]
dev_answer = [encoder.encode(sent) for sent in dev_answer]
test_question = [encoder.encode(sent) for sent in test_question]
test_answer = [encoder.encode(sent) for sent in test_answer]

# Padding sentences
max_question_len = 28
max_answer_len = 255
train_question, train_answer = data_helpers.padding_sent(train_question, train_answer, max_answer_len, max_question_len)
test_question, test_answer = data_helpers.padding_sent(test_question, test_answer, max_answer_len, max_question_len)
dev_question, dev_answer = data_helpers.padding_sent(dev_question, dev_answer, max_answer_len, max_question_len)

# List to numpy array
test_question = np.asarray(test_question)
test_answer = np.asarray(test_answer)
test_label = np.asarray(test_label)
test_x_feat = np.asarray(test_x_feat)

train_question = np.asarray(train_question)
train_answer = np.asarray(train_answer)
train_label = np.asarray(train_label)
train_x_feat = np.asarray(train_x_feat)

dev_question = np.asarray(dev_question)
dev_answer = np.asarray(dev_answer)
dev_label = np.asarray(dev_label)
dev_x_feat = np.asarray(dev_x_feat)

# Numpy array to Dataset
BUFFER_SIZE = 60000
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"input_question": train_question, "input_answer": train_answer, "input_x_feat": train_x_feat},
        {"outputs": train_label},
    )
)

train_dataset = train_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
train_dataset = train_dataset.batch(BATCH_SIZE)

dev_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"input_question": dev_question, "input_answer": dev_answer, "input_x_feat": dev_x_feat},
        {"outputs": dev_label},
    )
)
dev_dataset = dev_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
dev_dataset = dev_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"input_question": test_question, "input_answer": test_answer, "input_x_feat": test_x_feat},
        {"outputs": test_label},
    )
)
# test_dataset = test_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
test_dataset = test_dataset.batch(BATCH_SIZE)
