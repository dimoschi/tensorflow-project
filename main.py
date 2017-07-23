# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:03:13 2017

@author: apaterakis
"""

from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Parameters
learning_rate = 0.001
training_iters = 30000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 259*194*3  # data input (img shape)
n_classes = 2  # total classes (0-1: uncensored - censored)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, shape=[None, n_input])
y = tf.placeholder(tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# IMPORT IMAGES
def get_images(censored_counter, uncensored_counter):
    images = []
    y = []
    path_censored = os.path.join(os.getcwd(), "input", "train", "censored")
    path_uncensored = os.path.join(os.getcwd(), "input", "train", "uncensored")
    for i in range(uncensored_counter, uncensored_counter+2):
        images.append(Image.open(
            os.path.join(path_uncensored, "{}.jpg".format(i))))
        y.append([0, 1])
    images.append(Image.open(
        os.path.join(path_censored, "{}.jpg".format(censored_counter))))
    y.append([1, 0])
    return images, y


def get_test_images():
    images = []
    images_y = []
    censored = os.listdir(os.path.join(
        os.getcwd(), "input", "test", "censored")
    )
    uncensored = os.listdir(os.path.join(
        os.getcwd(), "input", "test", "uncensored")
    )
    for file in uncensored:
        images.append(Image.open(file))
        images_y.append([0, 1])
    for file in censored:
        images.append(Image.open(file))
        images_y.append([1, 0])
    return images, images_y


def get_batch(images, images_x):  # batch_size
    # images, y = get_images(batch_size)
    batch_x = np.asarray([np.asarray(
        image, dtype=np.float32).flatten() for image in images])
    batch_y = np.asarray(images_x)
    return batch_x, batch_y


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 3], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(
        x, ksize=[1, k, k, 3], strides=[1, k, k, 3], padding='SAME'
    )


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 194, 259, 3])
    #
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    #
    # Convolution Layer
    # conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    # conv2 = maxpool2d(conv2, k=2)
    # print(conv2.get_shape())
    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # print(conv3.get_shape())
    #
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
    # fc1 = tf.reshape(conv2, [-1, 7*7*64])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    #
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 32])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=pred, labels=y)
)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    censored_count = 1
    uncensored_count = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        import ipdb; ipdb.set_trace()
        images, classes = get_images(censored_count, uncensored_count)
        batch_x, batch_y = get_batch(images, classes)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={
            x: batch_x,
            y: batch_y,
            keep_prob: dropout
        })
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: 1.
            })
            # Calculate accuracy for 256 mnist test images
            test_images, test_y = get_test_images()
            test_x, test_y = get_batch(test_images, test_y)
            t_acc = sess.run(accuracy, feed_dict={
                x: test_x,
                y: test_y,
                keep_prob: 1.
            })
            print(
                "Iter " + str(step*batch_size) +
                ", Minibatch Loss= " + "{:.6f}".format(loss) +
                ", Training Accuracy= " + "{:.5f}".format(acc) +
                ", Testing Accuracy= " + "{:.5f}".format(t_acc)
            )
        censored_count += 1
        uncensored_count += 2
    print("Optimization Finished!")
