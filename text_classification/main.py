# https://sourcedexter.com/tensorflow-text-classification-python/
# http://tflearn.org/   - tflearn tutorial

import tflearn
import tensorflow as tf
from text_classification.utils import *
from text_classification.models import *
import sys
import unicodedata
from sklearn.model_selection import train_test_split

# from tensorflow.contrib import bayesflow
# from tensorflow.tensorboard.tensorboard import main
from tensorboard.plugins.projector.projector_plugin import PROJECTOR_FILENAME

#=========================================
data_path = "./data/"
model_path = "./model/"
data_file = "data_big_test.csv"
# data_file = "data_small_test.csv"
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
print ("Size of words = ", words.__len__())
print ("Size of docs = ", docs.__len__())
training = get_training(categories, docs, words)
print ("training set have been created")
# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print (len(train_x[0]))
print (len(train_y[0]))

# split into train and test
train_x, test_x, train_y, test_y = train_test_split( train_x, train_y, test_size=0.33, random_state=42)

# model = model_1(model_path + 'model.tflearn', train_x, train_y)
model = model_2(train_x, train_y, test_x, test_y, 100)
# model.load(model_path + 'model.tflearn')

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
