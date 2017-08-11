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
    "log_dir": "tslog",
    "run_name": "test_conv_1"
}


# tf Graph input
with tf.name_scope('set-nn-variables'):
    x = tf.placeholder(
        tf.float32,
        shape=[None, KEY_PARAMETERS["height"], KEY_PARAMETERS["width"], 3],
        name="Input_Images"
    )
    y = tf.placeholder(
        tf.float32, shape=[None, KEY_PARAMETERS["n_classes"]],
        name="Output_Classes"
    )
    # dropout (keep probability)
    keep_prob = tf.placeholder(tf.float32)


# IMPORT images
def split_dataset(ratio):
    with tf.name_scope('Split_dataset'):
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
    with tf.name_scope('Get_images'):
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
    with tf.name_scope('Fix_image'):
        required_size = KEY_PARAMETERS["required_size"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != required_size:
            image = image.resize(required_size, Image.LANCZOS)
    return image


def get_batch(images, y):
    with tf.name_scope('Create_batch'):
        batch_x = np.asarray([np.asarray(
            image, dtype=np.float32) for image in images])
        batch_y = np.asarray(y)
    return batch_x, batch_y


# Create some wrappers for simplicity
def conv2d(x, W, b, name, strides=1):
    # Conv2D wrapper, with bias and relu activation
    with tf.name_scope(name):
        x = tf.nn.conv2d(
            x, W, strides=[1, strides, strides, 1],
            padding='SAME', name="Convolution"
        )
        x = tf.nn.bias_add(x, b, name="Bias_Add")
        tf.summary.histogram('Pre_activation', x)
        x = tf.nn.relu(x, name="Relu_activation")
        tf.summary.histogram('Activation', x)
    return x


def maxpool2d(x, name, k=2):
    # MaxPool2D wrapper
    with tf.name_scope(name):
        max_pool = tf.nn.max_pool(
            x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
            padding='SAME', name="Max_Pooling"
        )
    return max_pool


def fully_connected(conv_layer, shape, w, b, name, do=1):
    with tf.name_scope(name):
        fc = tf.reshape(conv_layer, shape)
        fc = tf.add(tf.matmul(fc, w), b)
        fc = tf.nn.relu(fc)
        # Apply Dropout
        fc = tf.nn.dropout(fc, do)
    return fc


# Create model
def conv_net(x, weights, biases, dropout):
    # Convolution Layer #1
    with tf.name_scope('raw_image_layer'):
        V = tf.slice(
            x, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input'
        )
        tf.summary.image("raw_image", V)

    conv1 = conv2d(
        x, weights['wc1'], biases['bc1'], 'Convolution_1', strides=2
    )

    with tf.name_scope('convolution-1_visualization'):
        # Prepare for visualization
        # Take only convolutions of first image, discard convolutions
        # for other images.
        V = tf.slice(
            conv1, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input'
        )

        # Reorder so the channels are in the first dimension, x and y follow.
        V = tf.transpose(V, (0, 3, 1, 2))
        # Bring into shape expected by image_summary
        V = tf.reshape(V, (-1, 48, 64, 1))

        tf.summary.image("conv_1_image", V, max_outputs=4)

    conv2 = conv2d(
        conv1, weights['wc2'], biases['bc2'], 'Convolution_2', strides=1
    )

    with tf.name_scope('convolution_2_visualization'):
        # Prepare for visualization
        # Take only convolutions of first image, discard convolutions
        # for other images.
        V = tf.slice(
            conv2, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input'
        )

        # Reorder so the channels are in the first dimension, x and y follow.
        V = tf.transpose(V, (0, 3, 1, 2))
        # Bring into shape expected by image_summary
        V = tf.reshape(V, (-1, 48, 64, 1))

        tf.summary.image("conv_2_image", V, max_outputs=6)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, 'Max_Pool_1', k=2)
    print(conv2.get_shape())
    # # Convolution Layer #2
    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 'Convolution_3')
    # conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], 'Convolution_4')
    # # Max Pooling (down-sampling)
    # conv4 = maxpool2d(conv4, 'Max_Pool_2', k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = fully_connected(
        conv2, [-1, 24*32*24], weights['wd1'], biases['bd1'],
        'Fully_connected', dropout
    )
    # Output, class prediction
    with tf.name_scope('output'):
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
with tf.name_scope('set_weigths_and_biases'):
    weights = {
        # 24x32 conv, 3 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([4, 4, 3, 12]), name='weight_1'),
        # 12x16 conv, 3 input, 32 outputs
        'wc2': tf.Variable(tf.random_normal([2, 2, 12, 24]), name='weight_2'),
        # 3x4 conv, 32 inputs, 64 outputs
        'wc3': tf.Variable(tf.random_normal([3, 4, 64, 64]), name='weight_3'),
        # 2x2 conv, 64 inputs, 128 outputs
        'wc4': tf.Variable(tf.random_normal([2, 2, 64, 64]), name='weight_4'),
        # fully connected, 24*32*128 inputs, 1024 outputs
        'wd1': tf.Variable(
            tf.random_normal([24*32*24, 1024]), name='weight_fc'
        ),
        # 1024 inputs, 2 outputs (class prediction)
        'out': tf.Variable(
            tf.random_normal([1024, KEY_PARAMETERS["n_classes"]]),
            name='weight_out'
        )
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([12]), name='bias_1'),
        'bc2': tf.Variable(tf.random_normal([24]), name='bias_2'),
        'bc3': tf.Variable(tf.random_normal([64]), name='bias_3'),
        'bc4': tf.Variable(tf.random_normal([64]), name='bias_4'),
        'bd1': tf.Variable(tf.random_normal([1024]), name='bias_fc'),
        'out': tf.Variable(
            tf.random_normal([KEY_PARAMETERS["n_classes"]]),
            name='bias_out'
        )
    }


# Construct model
with tf.name_scope('prediction'):
    pred = conv_net(x, weights, biases, keep_prob)


# Define loss and optimizer
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=pred, labels=y, name='cost')
    )
    tf.summary.scalar('cost', cost)


with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=KEY_PARAMETERS["learning_rate"]
    ).minimize(cost)


# Evaluate model
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)


# Saver Class
saver = tf.train.Saver(max_to_keep=1)


# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(
        os.path.join(
            os.getcwd(), KEY_PARAMETERS["log_dir"],
            KEY_PARAMETERS["run_name"], "train"
        ), sess.graph
    )
    test_writer = tf.summary.FileWriter(
        os.path.join(
            os.getcwd(), KEY_PARAMETERS["log_dir"],
            KEY_PARAMETERS["run_name"], "test"
        ))
    tf.global_variables_initializer().run()
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
        print("{} batches to train with {} images".format(
            total_batches, train_iter)
        )
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
            summary, loss, acc = sess.run(
                [merged, cost, accuracy],
                feed_dict={
                    x: train_batch_x,
                    y: train_batch_y,
                    keep_prob: dropout
                }
            )
            avg_cost += loss / total_batches
            avg_train_acc += acc / total_batches
            total_step = epoch*total_batches
            # Print every 20 steps
            if (step % 20 == 0) or (step % total_batches == 0):
                train_writer.add_summary(summary, total_step+step)
                print(
                    "Epoch: " + str(epoch+1) +
                    ", Step: " + str(step) +
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
        test_step = 1
        print("{} batches to test with {} images".format(
            total_batches, test_iter)
        )
        while test_step * batch_size <= test_iter:
            test_images, test_y = get_images(test_images_dict, batch_size)
            test_batch_x, test_batch_y = get_batch(test_images, test_y)
            summary, t_acc = sess.run([merged, accuracy], feed_dict={
                x: test_batch_x,
                y: test_batch_y,
                keep_prob: 1.
            })
            total_step = epoch*total_batches
            test_writer.add_summary(summary, total_step+test_step)
            avg_test_acc += t_acc / total_batches
            test_step += 1
        end_time = datetime.datetime.now()
        dt = end_time - start_time

        # If accuracy is better than best_accuracy
        # update best_model and accuracy
        if (avg_test_acc > best_accuracy):
            save_time = datetime.datetime.now()
            saver.save(
                sess,
                os.path.join(os.getcwd(), 'models', 'best-model')
            )
            best_accuracy = avg_test_acc
            save_time = datetime.datetime.now() - save_time
            print(
                "Best Model Updated. Accuracy: {}".format(
                    str(avg_test_acc))
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
            ", Testing Accuracy: " + "{:.5f}".format(avg_test_acc)
        )
    print("Optimization Finished!")
