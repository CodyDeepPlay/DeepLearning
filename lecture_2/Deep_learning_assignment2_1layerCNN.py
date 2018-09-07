# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 21:16:41 2018

@author: Mingming
"""

# DEEP LEARNING ASSIGNMENT 2: build 1 layer CNN with tensor flow
# from Udacity deep learning class
# https://classroom.udacity.com/courses/ud730/lessons/6379031992/concepts/65959889480923

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

#%% load the saved pickle file from assignment 1
pickle_file = 'C:/Users/Mingming/Dropbox1/Research_and_work/Python_Mingming/Deep learning classes/Udacity online classes/lecture 1/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  

#%%  Reformat into a shape that is more adapted to the models we are going to train
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

"""
Training set (200000, 784) (200000, 10), 784 is the 28*28, the image pixels
Validation set (10000, 784) (10000, 10)
Test set (10000, 784) (10000, 10)
"""

#%%  
"""
PROBLEM
Turn the logistic regression example with SGD into a 1-hidden layer neural network with rectified linear units 
nn.relu() and 1024 hidden nodes. 
This model should improve your validation / test accuracy.
"""

def multilayer_perceptron(x, weights, biases, keep_prob):
    # multiply matrix "x" and matrix "weights['h1']", and then add matrix "biases['b1']"
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer  


n_hidden_1   = 1024 # number of the neurons in hidden layer
display_step = 100
batch_size   = 128

graph = tf.Graph()  # A TensorFlow computation, represented as a dataflow graph.
with graph.as_default():
  # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset) # create a constant tensor
  tf_test_dataset  = tf.constant(test_dataset)
  
  
  n_input    = image_size * image_size
  n_classes  = num_labels
  # generate a tensor variable maintains state in the graph across calls to run()
  weights = {'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),   # generate random values from a normal distribution
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))}
  
  biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1])),
          'out': tf.Variable(tf.random_normal([n_classes]))}
  keep_prob = tf.placeholder("float") # dropout rate in the layer of neural network

  # Training computation
  logits = multilayer_perceptron(tf_train_dataset, weights, biases, keep_prob) # output from the neural network
  # computes the mean of elements across dimensions of a tensor
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) # softmax cross entropy between logits (neural net output) and labels
 
  # Optimizer.
  #optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss) #Construct a new gradient descent optimizer.
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
          multilayer_perceptron(tf_valid_dataset, weights, biases, keep_prob))
  test_prediction = tf.nn.softmax(
          multilayer_perceptron(tf_test_dataset, weights, biases, keep_prob))



#%% then run the tf flow model
training_epochs = 3001

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) # conduct one-hot encoding, the highest score will get the prediction label
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for epoch in range(training_epochs):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (epoch * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data   = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.5}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
       
    # print out the results after every other 500 training was done.
    if (epoch % 500 == 0):
      print("Minibatch loss at epoch %d: %f" % (epoch, l)  )
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)  )
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval({keep_prob: 0.5}), valid_labels)  )
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval({keep_prob: 0.5}), test_labels)  ) 

"""
They all used to format the string
%s: place holder for a string
%d: place holder for a int number
%f: place holder for a float/double number
"""