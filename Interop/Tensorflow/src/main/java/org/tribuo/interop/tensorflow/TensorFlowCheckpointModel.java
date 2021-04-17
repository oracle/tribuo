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

import org.tensorflow.exceptions.TensorFlowException;
import org.tensorflow.proto.framework.GraphDef;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.provenance.ModelProvenance;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.Closeable;
import java.io.IOException;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * TensorFlow support is experimental, and may change without a major version bump.
 * <p>
 * This model encapsulates a simple model with an input feed dict,
 * and produces a single output tensor.
 * <p>
 * If the checkpoint is not available on construction or after deserialisation then the
 * model is uninitialised. Models can be initialised by calling {@link #initialize} or
 * {@link #setCheckpointDirectory}.
 * <p>
 * This model's serialized form stores the weights in the specified model checkpoint directory.
 * If you wish to convert it into a model which stores the weights inside the model then
 * call {@link #convertToNativeModel}.
 * <p>
 * It accepts an {@link ExampleTransformer} that converts an example's features into a {@link TensorMap}, and an
 * {@link OutputTransformer} that converts a {@link Tensor} into a {@link Prediction}.
 * <p>
 * The model's serialVersionUID is set to the major Tensorflow version number times 100.
 * <p>
 * N.B. Tensorflow support is experimental and may change without a major version bump.
 */
public class TensorFlowCheckpointModel<T extends Output<T>> extends TensorFlowModel<T> implements Closeable {
    private static final Logger logger = Logger.getLogger(TensorFlowCheckpointModel.class.getName());

    private static final long serialVersionUID = 200L;

    private String checkpointDirectory;

    private boolean initialized;

    TensorFlowCheckpointModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDMap, GraphDef graphDef, String checkpointDirectory, int batchSize, String initName, String outputName, ExampleTransformer<T> exampleTransformer, OutputTransformer<T> outputTransformer) {
        super(name, description, featureIDMap, outputIDMap, graphDef, batchSize, initName, outputName, exampleTransformer, outputTransformer);
        this.checkpointDirectory = checkpointDirectory;
        try {
            session.restore(checkpointDirectory);
            initialized = true;
        } catch (TensorFlowException e) {
            logger.log(Level.WARNING, "Failed to initialise model in directory " + checkpointDirectory, e);
        }
    }

    /**
     * Is this model initialized?
     * @return True if the model is ready to make predictions.
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Initializes the model.
     * <p>
     * Throws {@code TensorFlowException} if it failed to read the checkpoint.
     */
    public final void initialize() {
        session.restore(checkpointDirectory);
        initialized = true;
    }

    /**
     * Sets the checkpoint directory. If the directories are different
     * then it re-initializes the model.
     * <p>
     * Throws {@code TensorFlowException} if the model fails to re-initialize.
     * @param newCheckpointDirectory The new checkpoint directory.
     */
    public void setCheckpointDirectory(String newCheckpointDirectory) {
        if (!checkpointDirectory.equals(newCheckpointDirectory)) {
            checkpointDirectory = newCheckpointDirectory;
            initialize();
        }
    }

    /**
     * Gets the checkpoint directory this model loads from.
     * @return The checkpoint directory.
     */
    public String getCheckpointDirectory() {
        return checkpointDirectory;
    }

    /**
     * Creates a {@link TensorFlowNativeModel} version of this model.
     * @return A version of this model using Tribuo's native serialization mechanism.
     */
    public TensorFlowNativeModel<T> convertToNativeModel() {
        Map<String, TensorFlowUtil.TensorTuple> tensorMap = TensorFlowUtil.serialise(modelGraph,session);
        return new TensorFlowNativeModel<>(name, provenance, featureIDMap,
                outputIDInfo, modelGraph.toGraphDef(), tensorMap, batchSize, initName, outputName, exampleTransformer, outputTransformer);
    }

    @Override
    protected TensorFlowCheckpointModel<T> copy(String newName, ModelProvenance newProvenance) {
        return new TensorFlowCheckpointModel<>(newName,newProvenance,featureIDMap,outputIDInfo,modelGraph.toGraphDef(),checkpointDirectory,batchSize,initName,outputName,exampleTransformer,outputTransformer);
    }

    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        if (closed) {
            throw new IllegalStateException("Can't serialize a closed model, the state has gone.");
        }
        out.defaultWriteObject();
        byte[] modelBytes = modelGraph.toGraphDef().toByteArray();
        out.writeObject(modelBytes);
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        byte[] modelBytes = (byte[]) in.readObject();
        this.modelGraph = new Graph();
        this.modelGraph.importGraphDef(GraphDef.parseFrom(modelBytes));
        this.session = new Session(modelGraph);

        try {
            session.restore(checkpointDirectory);
            initialized = true;
        } catch (TensorFlowException e) {
            logger.log(Level.WARNING, "Failed to initialise model after deserialization, attempted to load from " + checkpointDirectory, e);
        }
    }
}
