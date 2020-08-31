import numpy as np

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

# Convert list to numpy array
test_label = np.asarray(test_label)
test_x_feat = np.asarray(test_x_feat)
train_label = np.asarray(train_label)
train_x_feat = np.asarray(train_x_feat)
dev_label = np.asarray(dev_label)
dev_x_feat = np.asarray(dev_x_feat)
