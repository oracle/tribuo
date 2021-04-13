/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.tensorflow;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.proto.framework.GraphDef;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.util.Map;

/**
 * This model encapsulates a TensorFlow model running in graph mode with a single tensor output.
 * <p>
 * It accepts an {@link ExampleTransformer} that converts an example's features into a set of {@link Tensor}s, and an
 * {@link OutputTransformer} that converts a {@link Tensor} into a {@link Prediction}.
 * <p>
 * The model's serialVersionUID is set to the major Tensorflow version number times 100.
 * <p>
 * N.B. Tensorflow support is experimental and may change without a major version bump.
 */
class TensorFlowNativeModel<T extends Output<T>> extends TensorFlowModel<T> {

    TensorFlowNativeModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDMap, GraphDef trainedGraphDef, Map<String, TensorFlowUtil.TensorTuple> tensorMap, int batchSize, String initName, String outputName, ExampleTransformer<T> exampleTransformer, OutputTransformer<T> outputTransformer) {
        super(name, description, featureIDMap, outputIDMap, trainedGraphDef, batchSize, initName, outputName, exampleTransformer, outputTransformer);
        // Initialises the parameters.
        session.runner().addTarget(initName).run();
        TensorFlowUtil.deserialise(session,tensorMap);
    }

    @Override
    protected TensorFlowNativeModel<T> copy(String newName, ModelProvenance newProvenance) {
        return new TensorFlowNativeModel<>(newName,newProvenance,featureIDMap,outputIDInfo,modelGraph.toGraphDef(), TensorFlowUtil.serialise(modelGraph,session),batchSize,initName,outputName,exampleTransformer,outputTransformer);
    }

    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        if (closed) {
            throw new IllegalStateException("Can't serialize a closed model, the state has gone.");
        }
        out.defaultWriteObject();
        byte[] modelBytes = modelGraph.toGraphDef().toByteArray();
        out.writeObject(modelBytes);
        Map<String, TensorFlowUtil.TensorTuple> tensorMap = TensorFlowUtil.serialise(modelGraph, session);
        out.writeObject(tensorMap);
    }

    @SuppressWarnings("unchecked") //deserialising a typed map
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        byte[] modelBytes = (byte[]) in.readObject();
        Map<String, TensorFlowUtil.TensorTuple> tensorMap = (Map<String, TensorFlowUtil.TensorTuple>) in.readObject();
        modelGraph = new Graph();
        modelGraph.importGraphDef(GraphDef.parseFrom(modelBytes));
        session = new Session(modelGraph);
        // Initialises the parameters.
        session.runner().addTarget(initName).run();
        TensorFlowUtil.deserialise(session,tensorMap);
    }
}
