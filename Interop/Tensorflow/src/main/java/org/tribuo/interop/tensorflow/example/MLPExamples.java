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
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.nn.Relu;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TFloat32;
import org.tribuo.Trainer;

/**
 * Static factory methods which produce Multi-Layer Perceptron architectures.
 */
public abstract class MLPExamples {

    private MLPExamples() {}

    /**
     * Builds an MLP which expects the supplied number of inputs, has hiddenSizes.length hidden layers, before
     * emitting numOutput outputs. Uses ReLU as the activation function for the hidden layers.
     * @param inputName The name of the input placeholder.
     * @param numFeatures The number of input features.
     * @param hiddenSizes The hidden layer sizes.
     * @param numOutputs The number of output dimensions.
     * @return A pair of a graph and the name of the output operation.
     */
    public static GraphDefTuple buildMLPGraph(String inputName, int numFeatures, int[] hiddenSizes, int numOutputs) {
        if (numFeatures < 1) {
            throw new IllegalArgumentException("Must have a positive number of features, found " + numFeatures);
        }
        if (numOutputs < 1) {
            throw new IllegalArgumentException("Must have a positive number of outputs, found " + numOutputs);
        }
        if (hiddenSizes.length < 1) {
            throw new IllegalArgumentException("Must supply a hidden layer dimension.");
        }
        for (int i = 0; i < hiddenSizes.length; i++) {
            if (hiddenSizes[i] < 1) {
                throw new IllegalArgumentException("Hidden dimensions must be positive, found " + hiddenSizes[i]);
            }
        }
        Graph graph = new Graph();

        Ops tf = Ops.create(graph);

        Glorot<TFloat32> initializer = new Glorot<>(VarianceScaling.Distribution.TRUNCATED_NORMAL, Trainer.DEFAULT_SEED);

        // Inputs
        Placeholder<TFloat32> input = tf.withName(inputName).placeholder(TFloat32.class,
                Placeholder.shape(Shape.of(-1, numFeatures)));

        Operand<TFloat32> prevOutput = input;
        long prevLayerSize = numFeatures;
        for (int i = 0; i < hiddenSizes.length; i++) {
            // Fully connected layer
            Variable<TFloat32> fcWeights = tf.variable(initializer.call(tf,tf.array(prevLayerSize,hiddenSizes[i]),TFloat32.class));
            Variable<TFloat32> fcBiases = tf.variable(tf.fill(tf.array(hiddenSizes[i]), tf.constant(0.1f)));
            Relu<TFloat32> relu = tf.nn.relu(tf.math.add(tf.linalg.matMul(prevOutput, fcWeights), fcBiases));

            // Setup next iteration
            prevLayerSize = hiddenSizes[i];
            prevOutput = relu;
        }

        // Fully connected layer
        Variable<TFloat32> outputWeights = tf.variable(initializer.call(tf,tf.array(prevLayerSize,numOutputs),TFloat32.class));
        Variable<TFloat32> outputBiases = tf.variable(tf.fill(tf.array(numOutputs), tf.constant(0.1f)));
        Add<TFloat32> output = tf.math.add(tf.linalg.matMul(prevOutput, outputWeights), outputBiases);

        // Extract the graph def and op names
        GraphDef graphDef = graph.toGraphDef();
        String outputName = output.op().name();

        // Close the graph
        graph.close();

        return new GraphDefTuple(graphDef, inputName, outputName);
    }
}
