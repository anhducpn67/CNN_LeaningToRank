import pickle

import gensim as gensim
import numpy as np

import load_data

vocab = load_data.vocab
vocab_size = load_data.vocab_size
encoder = load_data.encoder
word2vec_path = "C:/Users/AnhDuc/Desktop/Study/LabAIDANTE/cnn_text_classification/data/GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

embedding_matrix = np.zeros((vocab_size, 300))
embedding_matrix[0] = np.zeros(300, dtype='float32')
for elements in vocab:
    if elements in model:
        embedding_matrix[encoder.encode(elements)[0]] = list(model.get_vector(elements))
    else:
        embedding_matrix[encoder.encode(elements)[0]] = list(np.random.uniform(-0.25, 0.25, 300))

with open('embedding_matrix.pickle', 'wb') as handle:
    pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
