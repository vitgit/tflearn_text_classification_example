# https://sourcedexter.com/tensorflow-text-classification-python/
# http://tflearn.org/   - tflearn tutorial

import tflearn
import tensorflow as tf
from text_classification.utils import *
import sys
import unicodedata

# from tensorflow.contrib import bayesflow
# from tensorflow.tensorboard.tensorboard import main
from tensorboard.plugins.projector.projector_plugin import PROJECTOR_FILENAME

#=========================================
data_path = "./data/"
model_path = "./model/"
# data_file = "data_big_test.csv"
data_file = "data_small_test.csv"
test_file = "test1.txt"
#=========================================
data_file = data_path + data_file
test_file = data_path + test_file
# data is dict : {category, texts}
data = get_data_dict_from_csv(data_file)

# nltk.download('punkt')

categories = list(data.keys())

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

words, docs = get_words_n_docs(data, tbl)
print ("words and docs have been created")
training = get_training(categories, docs, words)
print ("training set have been created")
# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network with tflearn http://tflearn.org/getting_started/
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8, activation='relu')
net = tflearn.fully_connected(net, 8, activation='relu')
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save(model_path + 'model.tflearn')

test_text(test_file, model, categories, words)

# https://stackoverflow.com/questions/39289431/tflearn-evaluate-a-model
predictions = model.predict(train_x)
print (predictions)

accuracy = 0
for prediction, actual in zip(predictions, train_y):
    predicted_class = np.argmax(prediction)
    actual_class = np.argmax(actual)
    if(predicted_class == actual_class):
        accuracy+=1

accuracy = accuracy / len(train_y)
print(accuracy)

# to split into train and test ?
#tf.split_v(tf.random_shuffle(...
# or
# import sklearn.model_selection as sk
# X_train, X_test, y_train, y_test =
# sk.train_test_split(features,labels,test_size=0.33, random_state = 42)