
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import pad_sequences


def model_1(model_file, train_x, train_y):
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8, activation='relu')
    net = tflearn.fully_connected(net, 8, activation='relu')
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)

    # Define model and setup tensorboard
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    # Start training (apply gradient descent algorithm)
    model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
    model.save(model_file)
    return model

def model_2(train_x, train_y, test_x, test_y, embedding_size):
    # train_x = pad_sequences(train_x, maxlen=100, value=0.)
    # test_x = pad_sequences(test_x, maxlen=100, value=0.)

    out_dim = embedding_size # embedding size
    num_cat = len(train_y[0])

    network = input_data(shape=[None, len(train_x[0])], name='input')
    network = tflearn.embedding(network, input_dim=len(train_x[0]), output_dim=out_dim)  # input_dim - vocab size
    branch1 = conv_1d(network, out_dim, 3, padding='same', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, out_dim, 4, padding='same', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, out_dim, 5, padding='same', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.5)
    network = fully_connected(network, num_cat, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(train_x, train_y, n_epoch=5, shuffle=True, validation_set=(test_x, test_y), show_metric=True, batch_size=32)
    return model