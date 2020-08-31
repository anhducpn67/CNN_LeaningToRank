import pickle

import numpy as np
import tensorflow_datasets as tfds

import data_helpers
import load_data

# Load data from load_data.py
train_question = load_data.train_question
train_answer = load_data.train_answer
dev_question = load_data.dev_question
dev_answer = load_data.dev_answer
test_question = load_data.test_question
test_answer = load_data.test_answer

# Build word vocab
# w_vocab = set()
# w_vocab.update(data_helpers.get_vocab(train_question, train_answer))
# w_vocab.update(data_helpers.get_vocab(dev_question, dev_answer))
# w_vocab.update(data_helpers.get_vocab(test_question, test_answer))
# w_vocab = list(w_vocab)
# with open('w_vocab.pickle', 'wb') as handle:
#     pickle.dump(w_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('w_vocab.pickle', 'rb') as handle:
    w_vocab = pickle.load(handle)
w_vocab_size = len(w_vocab) + 1

# Encode sentences
encoder = tfds.features.text.TokenTextEncoder(w_vocab)
train_question = [encoder.encode(sent) for sent in train_question]
train_answer = [encoder.encode(sent) for sent in train_answer]
dev_question = [encoder.encode(sent) for sent in dev_question]
dev_answer = [encoder.encode(sent) for sent in dev_answer]
test_question = [encoder.encode(sent) for sent in test_question]
test_answer = [encoder.encode(sent) for sent in test_answer]

# Padding
max_question_len = 28
max_answer_len = 255
train_question, train_answer = data_helpers.padding_sent(train_question, train_answer, max_answer_len, max_question_len)
test_question, test_answer = data_helpers.padding_sent(test_question, test_answer, max_answer_len, max_question_len)
dev_question, dev_answer = data_helpers.padding_sent(dev_question, dev_answer, max_answer_len, max_question_len)

# List to numpy array
w_train_question = np.asarray(train_question)
w_train_answer = np.asarray(train_answer)
w_dev_question = np.asarray(dev_question)
w_dev_answer = np.asarray(dev_answer)
w_test_question = np.asarray(test_question)
w_test_answer = np.asarray(test_answer)
