import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import load_data
import tensorflow.keras.backend as K


# F1_score
def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# Load data
# train_question = load_data.train_question
# train_answer = load_data.train_answer
# train_label = load_data.train_label
# train_x_feat = load_data.train_x_feat

train_dataset = load_data.train_dataset
test_dataset = load_data.test_dataset
dev_dataset = load_data.dev_dataset
vocab_size = load_data.vocab_size

with open('embedding_matrix.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)

embedding_matrix = tf.keras.initializers.Constant(embedding_matrix)

# Build model
max_question_len = load_data.max_question_len
max_answer_len = load_data.max_answer_len

inputs_question = keras.Input(shape=(max_question_len,), name="input_question")
inputs_answer = keras.Input(shape=(max_answer_len,), name="input_answer")
inputs_x_feat = keras.Input(shape=(4,), name="input_x_feat")

embedding_dim = 300
embedding_layer = layers.Embedding(vocab_size, embedding_dim,
                                   embeddings_initializer=embedding_matrix,
                                   trainable=False, name="embedding_question")
embedding_layer_question = embedding_layer(inputs_question)
embedding_layer_answer = embedding_layer(inputs_answer)

conv_layer_question = layers.Conv1D(100, 5, activation='relu', name="filter_question", padding="same",
                                    kernel_regularizer=regularizers.l2(1e-5))(embedding_layer_question)
conv_layer_answer = layers.Conv1D(100, 5, activation='relu', name="filter_answer", padding="same",
                                  kernel_regularizer=regularizers.l2(1e-5))(embedding_layer_answer)

max_pool_layer_question = layers.MaxPool1D(pool_size=max_question_len, name="max_pool_question")(conv_layer_question)
max_pool_layer_answer = layers.MaxPool1D(pool_size=max_answer_len, name="max_pool_answer")(conv_layer_answer)

flatten_layer_question = layers.Flatten()(max_pool_layer_question)
flatten_layer_answer = layers.Flatten()(max_pool_layer_answer)

matrix_m = layers.Dense(100, use_bias=False, name="matrix_M")(flatten_layer_question)
sim = layers.Dot(axes=-1, name="sim")([matrix_m, flatten_layer_answer])

concatenate_layer = layers.concatenate([flatten_layer_question, sim, flatten_layer_answer, inputs_x_feat])
fc_layer = layers.Dense(205, kernel_regularizer=regularizers.l2(1e-4))(concatenate_layer)
drop_out_layer = layers.Dropout(0.5)(fc_layer)
outputs = layers.Dense(1, name="outputs")(drop_out_layer)

model = keras.Model(inputs=[inputs_question, inputs_answer, inputs_x_feat], outputs=[outputs], name="cnn_LTR")
keras.utils.plot_model(model, "model.png", show_shapes=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
# metrics=[keras.metrics.Precision(name='precision'),
#          keras.metrics.Recall(name='recall'),
#          'accuracy'])

model.fit(train_dataset, epochs=10,
          # validation_data=dev_dataset,
          # callbacks=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2, mode='min'),
          verbose=2, shuffle=True)

# model.evaluate(test_dataset, verbose=2)

test_info = load_data.test_info
dev_info = load_data.dev_info
predictions = model.predict(test_dataset)

# with open('predictions.pickle', 'wb') as handle:
#     pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('predictions.pickle', 'rb') as handle:
#     predictions = pickle.load(handle)

test_label = load_data.test_label


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
    # print(tmp)
    for idx in range(0, len(tmp)):
        if tmp[idx][1] == [1]:
            # print(idx)
            MRR = MRR + 1 / (idx + 1)
            break
    # print("\n")

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
