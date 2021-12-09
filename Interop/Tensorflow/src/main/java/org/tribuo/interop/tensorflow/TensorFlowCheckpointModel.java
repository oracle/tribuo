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
import java.nio.file.Paths;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This model encapsulates a simple model with an input feed dict,
 * and produces a single output tensor.
 * <p>
 * If the checkpoint is not available on construction or after deserialisation then the
 * model is uninitialised. Models can be initialised by calling {@link #initialize} after calling
 * {@link #setCheckpointDirectory} and {@link #setCheckpointName} with the right directory and name
 * respectively.
 * <p>
 * This model's serialized form stores the weights in the specified model checkpoint directory.
 * If you wish to convert it into a model which stores the weights inside the model then
 * call {@link #convertToNativeModel}.
 * <p>
 * It accepts an {@link FeatureConverter} that converts an example's features into a {@link TensorMap}, and an
 * {@link OutputConverter} that converts a {@link Tensor} into a {@link Prediction}.
 * <p>
 * The model's serialVersionUID is set to the major TensorFlow version number times 100.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 */
public final class TensorFlowCheckpointModel<T extends Output<T>> extends TensorFlowModel<T> implements Closeable {
    private static final Logger logger = Logger.getLogger(TensorFlowCheckpointModel.class.getName());

    private static final long serialVersionUID = 200L;

    private String checkpointDirectory;

    private String checkpointName;

    private boolean initialized;

    TensorFlowCheckpointModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDMap, GraphDef graphDef, String checkpointDirectory, String checkpointName, int batchSize, String outputName, FeatureConverter featureConverter, OutputConverter<T> outputConverter) {
        super(name, description, featureIDMap, outputIDMap, graphDef, batchSize, outputName, featureConverter, outputConverter);
        this.checkpointDirectory = checkpointDirectory;
        this.checkpointName = checkpointName;
        try {
            session.restore(resolvePath());
            initialized = true;
        } catch (TensorFlowException e) {
            logger.log(Level.WARNING, "Failed to initialise model in directory " + checkpointDirectory, e);
        }
    }

    /**
     * Resolves the path into the format that TensorFlow expects.
     * @return The TensorFlow checkpoint path.
     */
    private final String resolvePath() {
        return Paths.get(checkpointDirectory,checkpointName).toString();
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
     * This call closes the old session (if it exists) and creates a fresh session from the current checkpoint path.
     * <p>
     * Throws {@link org.tensorflow.exceptions.TensorFlowException} if it failed to read the checkpoint.
     */
    public final void initialize() {
        // Close the old session
        if (session != null) {
            session.close();
            session = null;
        }
        session = new Session(modelGraph);
        session.restore(resolvePath());

        initialized = true;
    }

    /**
     * Sets the checkpoint directory.
     * <p>
     * The model likely needs re-initializing after this call.
     * @param newCheckpointDirectory The new checkpoint directory.
     */
    public void setCheckpointDirectory(String newCheckpointDirectory) {
        checkpointDirectory = newCheckpointDirectory;
    }

    /**
     * Gets the checkpoint directory this model loads from.
     * @return The checkpoint directory.
     */
    public String getCheckpointDirectory() {
        return checkpointDirectory;
    }

    /**
     * Sets the checkpoint name.
     * <p>
     * The model likely needs re-initializing after this call.
     * @param newCheckpointName The new checkpoint name.
     */
    public void setCheckpointName(String newCheckpointName) {
        checkpointName = newCheckpointName;
    }

    /**
     * Gets the checkpoint name this model loads from.
     * @return The checkpoint name.
     */
    public String getCheckpointName() {
        return checkpointName;
    }

    /**
     * Creates a {@link TensorFlowNativeModel} version of this model.
     * @return A version of this model using Tribuo's native serialization mechanism.
     */
    public TensorFlowNativeModel<T> convertToNativeModel() {
        Map<String, TensorFlowUtil.TensorTuple> tensorMap = TensorFlowUtil.extractMarshalledVariables(modelGraph,session);
        return new TensorFlowNativeModel<>(name, provenance, featureIDMap,
                outputIDInfo, modelGraph.toGraphDef(), tensorMap, batchSize, outputName, featureConverter, outputConverter);
    }

    @Override
    protected TensorFlowCheckpointModel<T> copy(String newName, ModelProvenance newProvenance) {
        return new TensorFlowCheckpointModel<>(newName,newProvenance,featureIDMap,outputIDInfo,modelGraph.toGraphDef(),checkpointDirectory,checkpointName,batchSize,outputName, featureConverter, outputConverter);
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
            session.restore(resolvePath());
            initialized = true;
        } catch (TensorFlowException e) {
            logger.log(Level.WARNING, "Failed to initialise model after deserialization, attempted to load from " + checkpointDirectory, e);
        }
    }
}
