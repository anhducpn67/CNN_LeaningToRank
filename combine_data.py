import tensorflow as tf

import load_data
import preprocess_character_level
import preprocess_word_level

# Load label and x_feat from load_data
train_label = load_data.train_label
train_x_feat = load_data.train_x_feat
train_info = load_data.train_info

dev_label = load_data.dev_label
dev_x_feat = load_data.dev_x_feat
dev_info = load_data.dev_info

test_label = load_data.test_label
test_x_feat = load_data.test_x_feat
test_info = load_data.test_info

# Load word-based data from preprocess_word_level.py
w_train_question = preprocess_word_level.w_train_question
w_train_answer = preprocess_word_level.w_train_answer
w_dev_question = preprocess_word_level.w_dev_question
w_dev_answer = preprocess_word_level.w_dev_answer
w_test_question = preprocess_word_level.w_test_question
w_test_answer = preprocess_word_level.w_test_answer

# Load character-based data from preprocess_character_level.py
c_train_question = preprocess_character_level.c_train_question
c_train_answer = preprocess_character_level.c_train_answer
c_dev_question = preprocess_character_level.c_dev_question
c_dev_answer = preprocess_character_level.c_dev_answer
c_test_question = preprocess_character_level.c_test_question
c_test_answer = preprocess_character_level.c_test_answer

# Numpy array to Dataset
BUFFER_SIZE = 60000
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"w_input_question": w_train_question, "w_input_answer": w_train_answer,
         "c_input_question": c_train_question, "c_input_answer": c_train_answer,
         "input_x_feat": train_x_feat},
        {"outputs": train_label},
    )
)

train_dataset = train_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
train_dataset = train_dataset.batch(BATCH_SIZE)

dev_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"w_input_question": w_dev_question, "w_input_answer": w_dev_answer,
         "c_input_question": c_dev_question, "c_input_answer": c_dev_answer,
         "input_x_feat": dev_x_feat},
        {"outputs": dev_label},
    )
)
dev_dataset = dev_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
dev_dataset = dev_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"w_input_question": w_test_question, "w_input_answer": w_test_answer,
         "c_input_question": c_test_question, "c_input_answer": c_test_answer,
         "input_x_feat": test_x_feat},
        {"outputs": test_label},
    )
)
# test_dataset = test_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
test_dataset = test_dataset.batch(BATCH_SIZE)
