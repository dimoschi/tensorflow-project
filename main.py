# -*- coding: utf-8 -*-
"""This is the main file used in our tensorflow project"""

from __future__ import print_function

__author__ = "Antonis Paterakis, Dimosthenis Schizas"

import os
import random
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image

# Parameters
KEY_PARAMETERS = {
    "learning_rate": 0.001,
    "batch_size": 12,
    "display_step": 1,
    "required_size": (128, 96),  # tuple of required size
    "height": 96,
    "width": 128,  # data input (img shape)
    "n_classes": 2,  # total classes (0-1: uncensored - censored)
    "dropout": 0.75,  # Dropout, probability to keep units
    "censored_ratio": 0.4,
    "test_train_ratio": 0.2,
    "training_epochs": 10,
}


# tf Graph input
x = tf.placeholder(
    tf.float32,
    shape=[None, KEY_PARAMETERS["height"], KEY_PARAMETERS["width"], 3]
)
y = tf.placeholder(tf.float32, shape=[None, KEY_PARAMETERS["n_classes"]])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# IMPORT images
def split_dataset(ratio):
    test_images = dict()
    train_images = dict()
    censored_files = os.listdir(
        os.path.join(os.getcwd(), "input", "censored")
    )
    uncensored_files = os.listdir(
        os.path.join(os.getcwd(), "input", "uncensored")
    )

    test_images["censored"] = random.sample(
        censored_files, round(len(censored_files)*ratio)
    )
    train_images["censored"] = list(set(censored_files).difference(
        set(test_images["censored"])
    ))
    test_images["uncensored"] = random.sample(
        uncensored_files, round(len(uncensored_files)*ratio)
    )
    train_images["uncensored"] = list(set(uncensored_files).difference(
        set(test_images["uncensored"])
    ))

    return train_images, test_images


def get_images(images_dict, batch_size=None):
    images = []
    images_labels = []

    path_censored = os.path.join(os.getcwd(), "input", "censored")
    path_uncensored = os.path.join(os.getcwd(), "input", "uncensored")

    censored_ratio = KEY_PARAMETERS["censored_ratio"]
    try:
        censored_size = round(batch_size*censored_ratio)
        uncensored_size = batch_size - censored_size
    except:
        censored_size = len(images_dict["censored"])
        uncensored_size = len(images_dict["uncensored"])

    for image in random.sample(images_dict["uncensored"], uncensored_size):
        im = Image.open(os.path.join(path_uncensored, image))
        images.append(fix_image(im))
        images_labels.append([0, 1])
    for image in random.sample(images_dict["censored"], censored_size):
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
        image, dtype=np.float32) for image in images])
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
    # x = tf.reshape(x, shape=[-1, 220, 220, 3])
    # Convolution Layer #1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print(conv1.get_shape())
    # Convolution Layer #2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    # print(conv2.get_shape())
    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # print(conv3.get_shape())
    #
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, 48*32*64])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 12x16 conv, 3 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([12, 16, 3, 32])),
    # 3x4 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 4, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 32])),
    # fully connected, 24*32*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([48*32*64, 1024])),
    # 1024 inputs, 2 outputs (class prediction)
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

# Saver Class
saver = tf.train.Saver(max_to_keep=1)

# Launch the graph
with tf.Session() as sess:
    tf.logging.set_verbosity(tf.logging.DEBUG)
    sess.run(init)
    print("Session started")
    training_epochs = KEY_PARAMETERS["training_epochs"]
    test_train_ratio = KEY_PARAMETERS["test_train_ratio"]
    batch_size = KEY_PARAMETERS["batch_size"]
    dropout = KEY_PARAMETERS["dropout"]
    best_accuracy = 0
    for epoch in range(training_epochs):
        print("Epoch {} started".format(str(epoch+1)))
        start_time = datetime.datetime.now()
        step = 1
        avg_cost = 0.
        avg_train_acc = 0.
        avg_test_acc = 0.
        # Split dataset into train and test for current epoch
        train_images_dict, test_images_dict = split_dataset(test_train_ratio)
        train_iter = (
            len(train_images_dict["censored"]) +
            len(train_images_dict["uncensored"])
        )
        total_batches = int(train_iter/batch_size)
        print("Total batches to train: {}".format(total_batches))
        while step * batch_size <= train_iter:
            images, classes = get_images(train_images_dict, batch_size)
            train_batch_x, train_batch_y = get_batch(images, classes)
            # Run optimization op (backprop
            sess.run(optimizer, feed_dict={
                x: train_batch_x,
                y: train_batch_y,
                keep_prob: dropout
            })
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={
                x: train_batch_x,
                y: train_batch_y,
                keep_prob: dropout
            })
            avg_cost += loss / total_batches
            avg_train_acc += acc / total_batches
            # Print every 20 steps
            if (step % 20 == 0) or (step % total_batches == 0):
                print(
                    "Step: " + str(step) +
                    ", Average Cost: " + "{:.6f}".format(avg_cost) +
                    ", Average Training Accuracy: " +
                    "{:.6f}".format(avg_train_acc)
                )
            step += 1
        # Calculate accuracy for test images in batches
        test_iter = (
            len(test_images_dict["censored"]) +
            len(test_images_dict["uncensored"])
        )
        total_batches = int(test_iter/batch_size)
        while step * batch_size <= test_iter:
            test_images, test_y = get_images(test_images_dict)
            test_batch_x, test_batch_y = get_batch(test_images, test_y)
            t_acc = sess.run(accuracy, feed_dict={
                x: test_batch_x,
                y: test_batch_y,
                keep_prob: 1.
            })
            avg_train_acc += t_acc / total_batches
        end_time = datetime.datetime.now()
        dt = end_time - start_time

        # If accuracy is better than best_accuracy
        # update best_model and accuracy
        if (avg_train_acc > best_accuracy):
            save_time = datetime.datetime.now()
            saver.save(
                sess,
                os.path.join(os.getcwd(), 'models', 'best-model')
            )
            best_accuracy = avg_train_acc
            save_time = datetime.datetime.now() - save_time
            print(
                "Best Model Updated. Accuracy: {}".format(str(avg_train_acc))
            )
        else:
            print(
                "Best Model NOT Updated. (Best accuracy: {}".format(
                    str(best_accuracy)
                ))

        # Saving completed.

        print("Epoch {} run for {} minutes & {} seconds".format(
            str(epoch+1), dt.seconds // 60, dt.seconds % 60
        ))
        print(
            "Epoch: " + str(epoch+1) +
            ", Average Cost: " + "{:.6f}".format(avg_cost) +
            ", Training Accuracy: " + "{:.5f}".format(avg_train_acc) +
            ", Testing Accuracy: " + "{:.5f}".format(avg_train_acc)
        )
    print("Optimization Finished!")
