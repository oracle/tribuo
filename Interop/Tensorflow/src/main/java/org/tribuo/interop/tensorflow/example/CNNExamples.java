/*
 * Copyright (c) 2021 Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tribuo.interop.tensorflow.example;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.framework.initializers.Glorot;
import org.tensorflow.framework.initializers.VarianceScaling;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Concat;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.nn.Conv2d;
import org.tensorflow.op.nn.MaxPool;
import org.tensorflow.op.nn.Relu;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;
import org.tribuo.Trainer;

import java.util.Arrays;

/**
 * Static factory methods which produce Convolutional Neural Network architectures.
 */
public abstract class CNNExamples {

    /**
     * Private constructor for utility class.
     */
    private CNNExamples() {}

    /**
     * Builds a LeNet 5 style CNN (usually used for MNIST).
     * <p>
     * Expects there to only be a single colour pixel (i.e., a grayscale image).
     * Operates on square images.
     * <p>
     * Unlike the original LeNet 5 it uses ReLU activations.
     * @param inputName The input placeholder name.
     * @param imageSize The image width and height.
     * @param pixelDepth The maximum pixel value (usually 255).
     * @param numOutputs The number of output dimensions (usually 10 for MNIST).
     * @return A GraphRecord containing the LeNet graph and the relevant op names.
     */
    public static GraphDefTuple buildLeNetGraph(String inputName, int imageSize, int pixelDepth, int numOutputs) {
        if (imageSize < 1) {
            throw new IllegalArgumentException("Must have a positive image size, found " + imageSize);
        }
        if (pixelDepth < 1) {
            throw new IllegalArgumentException("Must have a positive pixel depth, found " + pixelDepth);
        }
        if (numOutputs < 1) {
            throw new IllegalArgumentException("Must have a positive number of outputs, found " + numOutputs);
        }
        final String PADDING_TYPE = "SAME";
        Graph graph = new Graph();

        Ops tf = Ops.create(graph);

        Glorot<TFloat32> initializer = new Glorot<>(VarianceScaling.Distribution.TRUNCATED_NORMAL, Trainer.DEFAULT_SEED);

        // Inputs
        Placeholder<TFloat32> input = tf.withName(inputName).placeholder(TFloat32.class,
                Placeholder.shape(Shape.of(-1, imageSize, imageSize, 1)));

        // Scaling the features
        Constant<TFloat32> centeringFactor = tf.constant(pixelDepth / 2.0f);
        Constant<TFloat32> scalingFactor = tf.constant((float) pixelDepth);
        Operand<TFloat32> scaledInput = tf.math.div(tf.math.sub(input, centeringFactor), scalingFactor);

        // First conv layer
        Variable<TFloat32> conv1Weights = tf.variable(initializer.call(tf, tf.array(5L, 5, 1, 32), TFloat32.class));
        Conv2d<TFloat32> conv1 = tf.nn.conv2d(scaledInput, conv1Weights, Arrays.asList(1L, 1L, 1L, 1L), PADDING_TYPE);
        Variable<TFloat32> conv1Biases = tf.variable(tf.fill(tf.array(32), tf.constant(0.0f)));
        Relu<TFloat32> relu1 = tf.nn.relu(tf.nn.biasAdd(conv1, conv1Biases));

        // First pooling layer
        MaxPool<TFloat32> pool1 = tf.nn
                .maxPool(relu1, tf.array(1, 2, 2, 1), tf.array(1, 2, 2, 1), PADDING_TYPE);

        // Second conv layer
        Variable<TFloat32> conv2Weights = tf.variable(initializer.call(tf, tf.array(5L, 5, 32, 64), TFloat32.class));
        Conv2d<TFloat32> conv2 = tf.nn.conv2d(pool1, conv2Weights, Arrays.asList(1L, 1L, 1L, 1L), PADDING_TYPE);
        Variable<TFloat32> conv2Biases = tf.variable(tf.fill(tf.array(64), tf.constant(0.1f)));
        Relu<TFloat32> relu2 = tf.nn.relu(tf.nn.biasAdd(conv2, conv2Biases));

        // Second pooling layer
        MaxPool<TFloat32> pool2 = tf.nn.maxPool(relu2, tf.array(1, 2, 2, 1), tf.array(1, 2, 2, 1),
                        PADDING_TYPE);

        // Compute the new shape
        long[] poolShape = pool2.shape().subShape(1,4).asArray();
        long numFlattenedFeatures = (long)(poolShape[0] * poolShape[1] * poolShape[2]);
        Concat<TInt64> newShape =  tf.concat(Arrays.asList(tf.array(-1L), tf.array(numFlattenedFeatures)),tf.constant(0));

        // Flatten inputs
        Reshape<TFloat32> flatten = tf.reshape(pool2, newShape);

        // Fully connected layer
        Variable<TFloat32> fc1Weights = tf.variable(initializer.call(tf, tf.concat(
                Arrays.asList(tf.array(numFlattenedFeatures), tf.array(512L)),tf.constant(0)),
                TFloat32.class));
        Variable<TFloat32> fc1Biases = tf.variable(tf.fill(tf.array(512), tf.constant(0.1f)));
        Relu<TFloat32> relu3 = tf.nn.relu(tf.math.add(tf.linalg.matMul(flatten, fc1Weights), fc1Biases));

        // Softmax layer
        Variable<TFloat32> fc2Weights = tf.variable(initializer.call(tf, tf.array(512L,numOutputs),TFloat32.class));
        Variable<TFloat32> fc2Biases = tf.variable(tf.fill(tf.array(numOutputs), tf.constant(0.1f)));

        Add<TFloat32> logits = tf.math.add(tf.linalg.matMul(relu3, fc2Weights), fc2Biases);

        // Extract the graph def and op names
        GraphDef graphDef = graph.toGraphDef();
        String outputName = logits.op().name();

        // Close the graph
        graph.close();

        return new GraphDefTuple(graphDef, inputName, outputName);
    }
}
