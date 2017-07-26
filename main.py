# -*- coding: utf-8 -*-
"""This is the main file used in our tensorflow project"""

from __future__ import print_function

__author__ = "Antonis Paterakis, Dimosthenis Schizas"

import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image

# Parameters
KEY_PARAMETERS = {
    "learning_rate": 0.001,
    "training_iters": 100,
    "batch_size": 6,
    "display_step": 1,
    "required_size": (259, 194),  # tuple of required size
    "n_input": 259*194*3,  # data input (img shape)
    "n_classes": 2,  # total classes (0-1: uncensored - censored)
    "dropout": 0.75,  # Dropout, probability to keep units
}


# tf Graph input
x = tf.placeholder(tf.float32, shape=[None, KEY_PARAMETERS["n_input"]])
y = tf.placeholder(tf.float32, shape=[None, KEY_PARAMETERS["n_classes"]])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# IMPORT images
# TODO: discard bad images
def get_images(batch_size=10, folder="train"):
    images = []
    images_labels = []

    path_censored = os.path.join(os.getcwd(), "input", folder, "censored")
    path_uncensored = os.path.join(os.getcwd(), "input", folder, "uncensored")

    uncensored = round(batch_size*0.8)
    censored = batch_size - uncensored
    for image in random.sample(os.listdir(path_uncensored), uncensored):
        im = Image.open(os.path.join(path_uncensored, image))
        images.append(fix_image(im))
        images_labels.append([0, 1])
    for image in random.sample(os.listdir(path_censored), censored):
        im = Image.open(os.path.join(path_censored, image))
        images.append(fix_image(im))
        images_labels.append([1, 0])
    return images, images_labels


def fix_image(image):
    """
    Check if an image needs to be resized and/or
    converted from grayscale to RGB

    Args:
        image: An image as PIL.Images

    Returns:
        image: An image as PIL.Image in specific size and in RGB
    """
    required_size = KEY_PARAMETERS["required_size"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != required_size:
        image = image.resize(required_size, Image.LANCZOS)
    return image


def get_batch(images, y):
    batch_x = np.asarray([np.asarray(
        image, dtype=np.float32).flatten() for image in images])
    batch_y = np.asarray(y)
    return batch_x, batch_y


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(
        x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME'
    )


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 194, 259, 3])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    # print(conv1.get_shape())
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
    # fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.reshape(conv1, [-1, 403520])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 7x2 conv, 3 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([2, 7, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 32])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([403520, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, KEY_PARAMETERS["n_classes"]]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([KEY_PARAMETERS["n_classes"]]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=pred, labels=y)
)
optimizer = tf.train.AdamOptimizer(
    learning_rate=KEY_PARAMETERS["learning_rate"]
).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step < 101:
        batch_size = KEY_PARAMETERS["batch_size"]
        dropout = KEY_PARAMETERS["dropout"]
        images, classes = get_images(batch_size, "train")
        train_batch_x, train_batch_y = get_batch(images, classes)
        # Run optimization op (backprop
        sess.run(optimizer, feed_dict={
            x: train_batch_x,
            y: train_batch_y,
            keep_prob: dropout
        })
        if step % 5 == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={
                x: train_batch_x,
                y: train_batch_y,
                keep_prob: dropout
            })
            # Calculate accuracy for 256 mnist test images
            test_images, test_y = get_images(20, "test")
            test_batch_x, test_batch_y = get_batch(test_images, test_y)
            t_acc = sess.run(accuracy, feed_dict={
                x: test_batch_x,
                y: test_batch_y,
                keep_prob: 1.
            })
            print(
                "Iter " + str(step) +
                ", Minibatch Loss= " + "{:.6f}".format(loss) +
                ", Training Accuracy= " + "{:.5f}".format(acc) +
                ", Testing Accuracy= " + "{:.5f}".format(t_acc)
            )
        step += 1
    print("Optimization Finished!")
