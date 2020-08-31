import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import preprocess_word_level
import preprocess_character_level
import combine_data

# Load data
train_dataset = combine_data.train_dataset
dev_dataset = combine_data.dev_dataset
test_dataset = combine_data.test_dataset

with open('w_embedding_matrix.pickle', 'rb') as handle:
    w_embedding_matrix = pickle.load(handle)
c_embedding_matrix = preprocess_character_level.c_embedding_matrix

w_embedding_matrix = tf.keras.initializers.Constant(w_embedding_matrix)
c_embedding_matrix = tf.keras.initializers.Constant(c_embedding_matrix)

w_vocab_size = preprocess_word_level.w_vocab_size
w_max_question_len = preprocess_word_level.max_question_len
w_max_answer_len = preprocess_word_level.max_answer_len

c_vocab_size = preprocess_character_level.c_vocab_size
c_max_question_len = preprocess_character_level.max_question_len
c_max_answer_len = preprocess_character_level.max_answer_len

# Build model

# Input
inputs_x_feat = keras.Input(shape=(4,), name="input_x_feat")

w_inputs_question = keras.Input(shape=(w_max_question_len,), name="w_input_question")
w_inputs_answer = keras.Input(shape=(w_max_answer_len,), name="w_input_answer")

c_inputs_question = keras.Input(shape=(c_max_question_len,), name="c_input_question")
c_inputs_answer = keras.Input(shape=(c_max_answer_len,), name="c_input_answer")

# Word-based embedding
w_embedding_dim = 300
w_embedding_layer = layers.Embedding(w_vocab_size, w_embedding_dim,
                                     embeddings_initializer=w_embedding_matrix,
                                     trainable=False, name="w_embedding")
w_embedding_layer_question = w_embedding_layer(w_inputs_question)
w_embedding_layer_answer = w_embedding_layer(w_inputs_answer)

w_conv_layer = layers.Conv1D(100, 5, activation='relu', name="w_filter", padding="same",
                             kernel_regularizer=regularizers.l2(1e-5))

w_conv_layer_question = w_conv_layer(w_embedding_layer_question)
w_conv_layer_answer = w_conv_layer(w_embedding_layer_answer)

w_max_pool_layer_question = layers.MaxPool1D(pool_size=w_max_question_len, name="w_max_pool_question")(
    w_conv_layer_question)
w_max_pool_layer_answer = layers.MaxPool1D(pool_size=w_max_answer_len, name="w_max_pool_answer")(w_conv_layer_answer)

w_flatten_layer_question = layers.Flatten()(w_max_pool_layer_question)
w_flatten_layer_answer = layers.Flatten()(w_max_pool_layer_answer)

w_matrix_m = layers.Dense(100, use_bias=False, name="w_matrix_M")(w_flatten_layer_question)
w_sim = layers.Dot(axes=-1, name="w_sim")([w_matrix_m, w_flatten_layer_answer])

# Character-based embedding
c_embedding_dim = 69
c_embedding_layer = layers.Embedding(c_vocab_size + 1, c_embedding_dim,
                                     # embeddings_initializer=c_embedding_matrix,
                                     trainable=True, name='c_embedding')
c_embedding_layer_question = c_embedding_layer(c_inputs_question)
c_embedding_layer_answer = c_embedding_layer(c_inputs_answer)

# One
# conv_layers = [[256, 7, 3],
#                [256, 7, 3],
#                [256, 3, -1],
#                [256, 3, -1],
#                [256, 3, -1],
#                [256, 3, 3]]
#
# x = c_embedding_layer_question
# for filter_num, filter_size, pooling_size in conv_layers:
#     x = layers.Conv1D(filter_num, filter_size, activation='relu')(x)
#     if pooling_size != -1:
#         x = layers.MaxPool1D(pool_size=pooling_size)(x)
# c_flatten_layer_question = layers.Flatten()(x)
#
# x = c_embedding_layer_answer
# for filter_num, filter_size, pooling_size in conv_layers:
#     x = layers.Conv1D(filter_num, filter_size, activation='relu')(x)
#     if pooling_size != -1:
#         x = layers.MaxPool1D(pool_size=pooling_size)(x)
# c_flatten_layer_answer = layers.Flatten()(x)

# Two
# conv_3_layer = layers.Conv1D(100, 3, activation='relu', name="filter_size_3")
# conv_4_layer = layers.Conv1D(100, 4, activation='relu', name="filter_size_4")
# conv_5_layer = layers.Conv1D(100, 5, activation='relu', name="filter_size_5")
#
# conv_3_layer_question = conv_3_layer(c_embedding_layer_question)
# conv_4_layer_question = conv_4_layer(c_embedding_layer_question)
# conv_5_layer_question = conv_5_layer(c_embedding_layer_question)
# max_pool_3_layer_question = layers.MaxPool1D(pool_size=c_max_question_len - 2)(conv_3_layer_question)
# max_pool_4_layer_question = layers.MaxPool1D(pool_size=c_max_question_len - 3)(conv_4_layer_question)
# max_pool_5_layer_question = layers.MaxPool1D(pool_size=c_max_question_len - 4)(conv_5_layer_question)
# flatten_3_layer_question = layers.Flatten()(max_pool_3_layer_question)
# flatten_4_layer_question = layers.Flatten()(max_pool_4_layer_question)
# flatten_5_layer_question = layers.Flatten()(max_pool_5_layer_question)
# c_concatenate_layer_question = layers.concatenate([flatten_3_layer_question, flatten_4_layer_question, flatten_5_layer_question])
#
# conv_3_layer_answer = conv_3_layer(c_embedding_layer_answer)
# conv_4_layer_answer = conv_4_layer(c_embedding_layer_answer)
# conv_5_layer_answer = conv_5_layer(c_embedding_layer_answer)
# max_pool_3_layer_answer = layers.MaxPool1D(pool_size=c_max_answer_len - 2)(conv_3_layer_answer)
# max_pool_4_layer_answer = layers.MaxPool1D(pool_size=c_max_answer_len - 3)(conv_4_layer_answer)
# max_pool_5_layer_answer = layers.MaxPool1D(pool_size=c_max_answer_len - 4)(conv_5_layer_answer)
# flatten_3_layer_answer = layers.Flatten()(max_pool_3_layer_answer)
# flatten_4_layer_answer = layers.Flatten()(max_pool_4_layer_answer)
# flatten_5_layer_answer = layers.Flatten()(max_pool_5_layer_answer)
# c_concatenate_layer_answer = layers.concatenate([flatten_3_layer_answer, flatten_4_layer_answer, flatten_5_layer_answer])
#
# c_matrix_m = layers.Dense(300, use_bias=False, name="c_matrix_M")(c_concatenate_layer_question)
# c_sim = layers.Dot(axes=-1, name="c_sim")([c_matrix_m, c_concatenate_layer_answer])

# Three
c_conv_layer = layers.Conv1D(100, 5, activation='relu', name="c_filter")
c_conv_layer_question = c_conv_layer(c_embedding_layer_question)
c_conv_layer_answer = c_conv_layer(c_embedding_layer_answer)

c_max_pool_layer_question = layers.MaxPool1D(pool_size=c_max_question_len - 4)(c_conv_layer_question)
c_max_pool_layer_answer = layers.MaxPool1D(pool_size=c_max_answer_len - 4)(c_conv_layer_answer)

c_flatten_layer_question = layers.Flatten()(c_max_pool_layer_question)
c_flatten_layer_answer = layers.Flatten()(c_max_pool_layer_answer)

c_matrix_m = layers.Dense(100, use_bias=False, name="c_matrix_M")(c_flatten_layer_question)
c_sim = layers.Dot(axes=-1, name="c_sim")([c_matrix_m, c_flatten_layer_answer])

# FC layer
concatenate_layer = layers.concatenate([w_flatten_layer_question, w_sim, w_flatten_layer_answer,
                                        c_flatten_layer_question, c_sim, c_flatten_layer_answer,
                                        inputs_x_feat])
fc_layer = layers.Dense(205, kernel_regularizer=regularizers.l2(1e-4))(concatenate_layer)
drop_out_layer = layers.Dropout(0.5)(fc_layer)
outputs = layers.Dense(1, name="outputs")(drop_out_layer)

model = keras.Model(inputs=[w_inputs_question, w_inputs_answer,
                            c_inputs_question, c_inputs_answer,
                            inputs_x_feat],
                    outputs=[outputs], name="cnn_LTR")
keras.utils.plot_model(model, "model.png", show_shapes=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

for turns in range(0, 10):
    model.fit(train_dataset, epochs=20,
              validation_data=dev_dataset,
              callbacks=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2,
                                                      mode='min'),
              verbose=2, shuffle=True)

    # model.evaluate(test_dataset, verbose=2)

    test_info = combine_data.test_info
    dev_info = combine_data.dev_info
    predictions = model.predict(test_dataset)

    # with open('predictions.pickle', 'wb') as handle:
    #     pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('predictions.pickle', 'rb') as handle:
    #     predictions = pickle.load(handle)

    test_label = combine_data.test_label


    def take_first(elem):
        return elem[0]


    # Calculation MRR
    Start = 0
    MRR = 0
    for rgn in test_info:
        tmp = []
        for idx in range(Start, Start + rgn):
            tmp.append([predictions[idx], test_label[idx]])
        Start = Start + rgn
        tmp.sort(key=take_first, reverse=True)
        for idx in range(0, len(tmp)):
            if tmp[idx][1] == [1]:
                MRR = MRR + 1 / (idx + 1)
                break

    MRR = MRR * (1 / len(test_info))
    print(MRR)

    # Calculation mAP
    Start = 0
    mAP = 0
    for rgn in test_info:
        tmp = []
        for idx in range(Start, Start + rgn):
            tmp.append([predictions[idx], test_label[idx]])
        Start = Start + rgn
        tmp.sort(key=take_first, reverse=True)
        Precision = 0
        count_pos_answer = 0
        for idx in range(0, len(tmp)):
            if tmp[idx][1] == [1]:
                count_pos_answer += 1
                Precision = Precision + (count_pos_answer / (idx + 1))
        mAP += (Precision / count_pos_answer)

    mAP = mAP * (1 / len(test_info))
    print(mAP)
    print("\n")
