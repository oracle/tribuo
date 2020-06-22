/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * TensorFlow support is experimental, and may change without a major version bump.
 * <p>
 * This model encapsulates a simple model with a single input tensor (labelled {@link TensorflowModel#INPUT_NAME}),
 * and produces a single output tensor (labelled {@link TensorflowModel#OUTPUT_NAME}).
 * <p>
 * It accepts an {@link ExampleTransformer} that converts an example's features into a {@link Tensor}, and an
 * {@link OutputTransformer} that converts a {@link Tensor} into a {@link Prediction}.
 * <p>
 * The model's serialVersionUID is set to the major Tensorflow version number times 100.
 * <p>
 * N.B. Tensorflow support is experimental and may change without a major version bump.
 */
public class TensorflowCheckpointModel<T extends Output<T>> extends Model<T> implements Closeable {

    private static final long serialVersionUID = 100L;

    private transient Graph modelGraph = null;

    private transient Session session = null;

    private final String checkpointDirectory;

    private final ExampleTransformer<T> exampleTransformer;

    private final OutputTransformer<T> outputTransformer;

    TensorflowCheckpointModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDMap, byte[] graphDef, String checkpointDirectory, ExampleTransformer<T> exampleTransformer, OutputTransformer<T> outputTransformer) {
        super(name, description, featureIDMap, outputIDMap, outputTransformer.generatesProbabilities());
        this.exampleTransformer = exampleTransformer;
        this.outputTransformer = outputTransformer;
        this.checkpointDirectory = checkpointDirectory;
        this.modelGraph = new Graph();
        this.modelGraph.importGraphDef(graphDef);
        this.session = new Session(modelGraph);

        try (Tensor<String> checkpointPrefix = Tensors.create(Paths.get(checkpointDirectory+"/"+TensorflowCheckpointTrainer.MODEL_FILENAME).toString())) {
            // Initialises the parameters.
            session.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
        }
    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        // This adds overhead and triggers lookups for each feature, but is necessary to correctly calculate
        // the number of features used in this example.
        SparseVector vec = SparseVector.createSparseVector(example,featureIDMap,false);
        try (Tensor<?> transformedInput = exampleTransformer.transform(example,featureIDMap);
             Tensor<?> isTraining = Tensor.create(false);
             Tensor<?> outputTensor = session.runner()
                     .feed(TensorflowModel.INPUT_NAME,transformedInput)
                     .feed(TensorflowTrainer.IS_TRAINING,isTraining)
                     .fetch(TensorflowModel.OUTPUT_NAME).run().get(0)) {
            // Transform the returned tensor into a Prediction.
            return outputTransformer.transformToPrediction(outputTensor,outputIDInfo,vec.numActiveElements(),example);
        }
    }

    /**
     * Deep learning models don't do feature rankings. Use an Explainer.
     * <p>
     * This method always returns the empty map.
     * @param n the number of features to return.
     * @return The empty map.
     */
    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    /**
     * Deep learning models don't do excuses. Use an Explainer.
     * <p>
     * This method always returns {@link Optional#empty}.
     * @param example The input example.
     * @return {@link Optional#empty}.
     */
    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        return Optional.empty();
    }

    @Override
    protected TensorflowCheckpointModel<T> copy(String newName, ModelProvenance newProvenance) {
        return new TensorflowCheckpointModel<>(newName,newProvenance,featureIDMap,outputIDInfo,modelGraph.toGraphDef(),checkpointDirectory,exampleTransformer,outputTransformer);
    }

    @Override
    public void close() {
        if (session != null) {
            session.close();
        }
        if (modelGraph != null) {
            modelGraph.close();
        }
    }

    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        byte[] modelBytes = modelGraph.toGraphDef();
        out.writeObject(modelBytes);
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        byte[] modelBytes = (byte[]) in.readObject();
        this.modelGraph = new Graph();
        this.modelGraph.importGraphDef(modelBytes);
        this.session = new Session(modelGraph);

        try (Tensor<String> checkpointPrefix = Tensors.create(Paths.get(checkpointDirectory+"/"+TensorflowCheckpointTrainer.MODEL_FILENAME).toString())) {
            // Initialises the parameters.
            session.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
        }
    }
}
