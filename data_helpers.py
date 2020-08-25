import re

import numpy as np
import tensorflow_datasets as tfds
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

en_stops = set(stopwords.words('english'))


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_from_file(file_path):
    text_data = list(open(file_path, "r", encoding='utf-8').readlines())
    question = []
    answer_pos = []
    answer_neg = []
    is_question = is_answer_pos = is_answer_neg = False
    idx = -1
    for line in text_data:
        if is_question:
            is_question = False
            question.append(clean_str(line))
            idx += 1
            answer_neg.append([])
            answer_pos.append([])

        if is_answer_pos:
            answer_pos[idx].append(clean_str(line))
            is_answer_pos = False

        if is_answer_neg:
            answer_neg[idx].append(clean_str(line))
            is_answer_neg = False

        if "<question>" in line:
            is_question = True
        if "<positive>" in line:
            is_answer_pos = True
        if "<negative>" in line:
            is_answer_neg = True

    return question, answer_pos, answer_neg


def create_data(file_path):
    data_question, data_answer_pos, data_answer_neg = load_from_file(file_path)
    question = []
    answer = []
    label = []
    info = []
    num_data = len(data_question)
    for idx in range(0, num_data):
        if len(data_answer_pos[idx]) == 0:
            continue
        for sent in data_answer_pos[idx]:
            question.append(data_question[idx])
            answer.append(sent)
            label.append([1])
        for sent in data_answer_neg[idx]:
            question.append(data_question[idx])
            answer.append(sent)
            label.append([0])
        info.append(len(data_answer_pos[idx]) + len(data_answer_neg[idx]))
    return question, answer, label, info


def get_vocab(question, answer):
    vocab = set()
    tokenizer = tfds.features.text.Tokenizer()
    for sent in question:
        some_tokens = tokenizer.tokenize(sent)
        vocab.update(some_tokens)
    for sent in answer:
        some_tokens = tokenizer.tokenize(sent)
        vocab.update(some_tokens)
    return vocab


def padding_sent(question, answer, max_answer_len, max_question_len):
    for idx in range(0, len(question)):
        answer[idx] = answer[idx] + [0] * (max_answer_len - len(answer[idx]))
        question[idx] = question[idx] + [0] * (max_question_len - len(question[idx]))
    return question, answer


# def oversampling(question, answer, label):
#     over_question = []
#     over_answer = []
#     over_label = []
#     num_pos = num_neg = 0
#     for idx in range(0, len(question)):
#         if label[idx] == [0, 1]:
#             num_pos += 1
#         else:
#             num_neg += 1
#     num_repeat = (num_neg // num_pos) + 1
#     for idx in range(0, len(question)):
#         if num_pos == num_neg:
#             break
#         if list(label[idx]) == [0, 1]:
#             for repeat in range(0, num_repeat):
#                 over_question.append(question[idx])
#                 over_answer.append(answer[idx])
#                 over_label.append(label[idx])
#                 num_pos += 1
#                 if num_pos == num_neg:
#                     break
#     question = question + over_question
#     answer = answer + over_answer
#     label = label + over_label
#     return question, answer, label


def under_sampling(question, answer, label):
    under_question = []
    under_answer = []
    under_label = []
    num_pos = num_neg = 0
    for idx in range(0, len(label)):
        if label[idx] == [1]:
            num_pos += 1

    for idx in range(0, len(question)):
        if label[idx] == [1]:
            under_question.append(question[idx])
            under_answer.append(answer[idx])
            under_label.append(label[idx])
        if label[idx] == [0] and num_neg < num_pos:
            under_question.append(question[idx])
            under_answer.append(answer[idx])
            under_label.append(label[idx])
            num_neg += 1

    return under_question, under_answer, under_label


def shuffle_data(question, answer, label, x_feat):
    shuffle_indices = np.random.permutation(np.arange(len(question)))
    shuffle_question = []
    shuffle_answer = []
    shuffle_label = []
    shuffle_x_feat = []
    for idx in shuffle_indices:
        shuffle_question.append(question[idx])
        shuffle_answer.append(answer[idx])
        shuffle_label.append(label[idx])
        shuffle_x_feat.append(x_feat[idx])
    return shuffle_question, shuffle_answer, shuffle_label, shuffle_x_feat


def get_x_feat(question, answer):
    x_feat = []
    for idx in range(0, len(question)):
        corpus = [question[idx], answer[idx]]
        vectorizer = TfidfVectorizer()
        score = vectorizer.fit_transform(corpus).toarray()
        words_list = vectorizer.get_feature_names()
        all_words = nonstop_words = 0
        score_all_words = score_nonstop_words = 0
        # question_score_all_words = []
        # answer_score_all_words = []
        # question_score_nonstop_words = []
        # answer_score_nonstop_words = []
        for pos in range(0, len(words_list)):
            word = words_list[pos]
            if word in question[idx] and word in answer[idx]:
                all_words += 1
                score_all_words += score[1][pos]
                # question_score_all_words.append(score[0][pos])
                # answer_score_all_words.append(score[1][pos])
                if word not in en_stops:
                    nonstop_words += 1
                    score_nonstop_words += score[1][pos]
                    # question_score_nonstop_words.append(score[0][pos])
                    # answer_score_nonstop_words.append(score[1][pos])
        # if len(question_score_all_words) == 0:
        #     score_all_words = 0
        # else:
        #     score_all_words = 1 - cosine(question_score_all_words, answer_score_all_words)
        #
        # if len(question_score_nonstop_words) == 0:
        #     score_nonstop_words = 0
        # else:
        #     score_nonstop_words = 1 - cosine(question_score_nonstop_words, answer_score_nonstop_words)

        x_feat.append([all_words, score_all_words, nonstop_words, score_nonstop_words])
    return x_feat
