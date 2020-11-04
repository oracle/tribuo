# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Copyright (c) 2020, Oracle and/or its affiliates. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

Hacked on to emit a model that can be consumed by Tribuo.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.


def main():
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(tf.float32, name="input", shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int32, name="target", shape=(None,))

    train_data_scaled = (train_data_node - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

    is_training = tf.placeholder(tf.bool, name='is_training')

    # The Model definition
    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.global_variables_initializer().run()}
    # 2D convolution, with 'SAME' padding (i.e., the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].

    # Layer 1 - convolutional
    conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                                    stddev=0.1, seed=SEED, dtype=tf.float32), name="conv1-weights")
    conv1 = tf.nn.conv2d(train_data_scaled, conv1_weights, strides=[1, 1, 1, 1], padding='SAME', name="conv1")
    # Bias and rectified linear non-linearity.
    conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32), name="conv1-biases")
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases), name="conv1-relu")
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

    # Layer 2 - convolutional
    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=tf.float32), name="conv2-weights")
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME', name="conv2")
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name="conv2-biases")
    relu_2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases), name="conv2-relu")
    pool_2 = tf.nn.max_pool(relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = tf.shape(pool_2) # pool_2.get_shape().as_list()
    reshape = tf.reshape(pool_2, [pool_shape[0], -1], name="reshape")

    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    # Layer 3 - fully connected, depth 512.
    fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                              stddev=0.1, seed=SEED, dtype=tf.float32), name="fc1-weights")
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name="fc1-biases")
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases, name="hidden-layer")
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    hidden_d = tf.cond(is_training, lambda: tf.nn.dropout(hidden, 0.5, seed=SEED), lambda: hidden, name="dropout-cond")
    # Training computation: logits + cross-entropy loss.
    # Layer 4 - softmax
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED, dtype=tf.float32), name="fc2-weights")
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32), name="fc2-biases")
    logits = tf.add(tf.matmul(hidden_d, fc2_weights), fc2_biases, name="logits")
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=logits), name="first-loss")

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss = tf.add(loss, 5e-4 * regularizers, name="training_loss")

    # Optimizer: insert the epoch count
    epoch = tf.placeholder(tf.int32, name="epoch")
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(learning_rate=0.01,  # Base learning rate.
                                               global_step=epoch,  # Current index into the dataset.
                                               decay_steps=1,  # Decay step.
                                               decay_rate=0.95,  # Decay rate.
                                               staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, name="train")

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits, name="output")

    init = tf.global_variables_initializer()

    saver_def = tf.train.Saver().as_saver_def()

    print('Operation to initialize variables:       ', init.name)
    print('Tensor to feed as input data:            ', train_data_node.name)
    print('Tensor to feed as training targets:      ', train_labels_node.name)
    print('Tensor to fetch as prediction:           ', train_prediction.name)
    print('Operation to train one step:             ', optimizer.name)
    print('Training loss name:                      ', loss.name)
    print('Tensor to be fed for checkpoint filename:', saver_def.filename_tensor_name)
    print('Operation to save a checkpoint:          ', saver_def.save_tensor_name)
    print('Operation to restore a checkpoint:       ', saver_def.restore_op_name)

    with open('graph.pb', 'w') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())


if __name__ == "__main__":
    main()
