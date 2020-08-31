import numpy as np
import tensorflow as tf

import load_data

# Build character vocab
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
tk = tf.keras.preprocessing.text.Tokenizer(num_words=None, char_level=True, oov_token='UNK')
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1
tk.word_index = char_dict.copy()
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
c_vocab_size = 69

# Load data from load_data.py
train_question = load_data.train_question
train_answer = load_data.train_answer
dev_question = load_data.dev_question
dev_answer = load_data.dev_answer
test_question = load_data.test_question
test_answer = load_data.test_answer

# Encode sentences
char_train_question = tk.texts_to_sequences(train_question)
char_train_answer = tk.texts_to_sequences(train_answer)
char_dev_question = tk.texts_to_sequences(dev_question)
char_dev_answer = tk.texts_to_sequences(dev_answer)
char_test_question = tk.texts_to_sequences(test_question)
char_test_answer = tk.texts_to_sequences(test_answer)

# Padding
max_question_len = 181
max_answer_len = 1475

char_train_question = tf.keras.preprocessing.sequence.pad_sequences(char_train_question,
                                                                    maxlen=max_question_len, padding='post')
char_train_answer = tf.keras.preprocessing.sequence.pad_sequences(char_train_answer,
                                                                  maxlen=max_answer_len, padding='post')
char_dev_question = tf.keras.preprocessing.sequence.pad_sequences(char_dev_question,
                                                                  maxlen=max_question_len, padding='post')
char_dev_answer = tf.keras.preprocessing.sequence.pad_sequences(char_dev_answer,
                                                                maxlen=max_answer_len, padding='post')
char_test_question = tf.keras.preprocessing.sequence.pad_sequences(char_test_question,
                                                                   maxlen=max_question_len, padding='post')
char_test_answer = tf.keras.preprocessing.sequence.pad_sequences(char_test_answer,
                                                                 maxlen=max_answer_len, padding='post')

# List to numpy array
c_train_question = np.array(char_train_question)
c_train_answer = np.array(char_train_answer)
c_dev_question = np.array(char_dev_question)
c_dev_answer = np.array(char_dev_answer)
c_test_question = np.array(char_test_question)
c_test_answer = np.array(char_test_answer)

# Character embedding matrix
c_embedding_matrix = np.zeros((70, 69))
c_embedding_matrix[0] = np.zeros(69, dtype='float64')
for char, i in tk.word_index.items():
    c_embedding_matrix[i][i - 1] = 1
c_embedding_matrix[69] = np.zeros(69, dtype='float64')
