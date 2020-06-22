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

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;

/**
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
public class TensorflowModel<T extends Output<T>> extends Model<T> implements Closeable {

    private static final Logger logger = Logger.getLogger(TensorflowModel.class.getName());

    private static final long serialVersionUID = 100L;

    public static final String INPUT_NAME = "input";
    public static final String OUTPUT_NAME = "output";

    private transient Graph modelGraph = null;

    private transient Session session = null;

    private int batchSize;

    private final ExampleTransformer<T> exampleTransformer;

    private final OutputTransformer<T> outputTransformer;

    TensorflowModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDMap, byte[] trainedGraphDef, Map<String, Object> tensorMap, int batchSize, ExampleTransformer<T> exampleTransformer, OutputTransformer<T> outputTransformer) {
        super(name, description, featureIDMap, outputIDMap, outputTransformer.generatesProbabilities());
        this.exampleTransformer = exampleTransformer;
        this.outputTransformer = outputTransformer;
        this.modelGraph = new Graph();
        this.modelGraph.importGraphDef(trainedGraphDef);
        this.session = new Session(modelGraph);
        this.batchSize = batchSize;
        // Initialises the parameters.
        session.runner().addTarget(TensorflowTrainer.INIT).run();
        TensorflowUtil.deserialise(session,tensorMap);
    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        // This adds overhead and triggers lookups for each feature, but is necessary to correctly calculate
        // the number of features used in this example.
        SparseVector vec = SparseVector.createSparseVector(example,featureIDMap,false);
        try (Tensor<?> transformedInput = exampleTransformer.transform(vec);
             Tensor<?> isTraining = Tensor.create(false);
             Tensor<?> outputTensor = session.runner()
                     .feed(INPUT_NAME,transformedInput)
                     .feed(TensorflowTrainer.IS_TRAINING,isTraining)
                     .fetch(OUTPUT_NAME).run().get(0)) {
            // Transform the returned tensor into a Prediction.
            return outputTransformer.transformToPrediction(outputTensor,outputIDInfo,vec.numActiveElements(),example);
        }
    }

    @Override
    protected List<Prediction<T>> innerPredict(Iterable<Example<T>> examples) {
        List<Prediction<T>> predictions = new ArrayList<>();
        List<Example<T>> batchExamples = new ArrayList<>();
        for (Example<T> example : examples) {
            batchExamples.add(example);
            if (batchExamples.size() == batchSize) {
                predictions.addAll(predictBatch(batchExamples));
                // clear the batch
                batchExamples.clear();
            }
        }

        if (!batchExamples.isEmpty()) {
            // send the partial batch
            predictions.addAll(predictBatch(batchExamples));
        }
        return predictions;
    }

    private List<Prediction<T>> predictBatch(List<Example<T>> batchExamples) {
        // Convert the batch
        List<SparseVector> vectors = new ArrayList<>(batchExamples.size());
        int[] numActiveElements = new int[batchExamples.size()];
        for (int i = 0; i < batchExamples.size(); i++) {
            SparseVector vec = SparseVector.createSparseVector(batchExamples.get(i),featureIDMap,false);
            numActiveElements[i] = vec.numActiveElements();
            vectors.add(vec);
        }

        // Send a batch to Tensorflow
        try (Tensor<?> transformedInput = exampleTransformer.transform(vectors);
             Tensor<?> isTraining = Tensor.create(false);
             Tensor<?> outputTensor = session.runner()
                     .feed(INPUT_NAME,transformedInput)
                     .feed(TensorflowTrainer.IS_TRAINING,isTraining)
                     .fetch(OUTPUT_NAME).run().get(0)) {
            // Transform the returned tensor into a list of Predictions.
            return outputTransformer.transformToBatchPrediction(outputTensor,outputIDInfo,numActiveElements,batchExamples);
        }
    }

    /**
     * Gets the current testing batch size.
     * @return The batch size.
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Sets a new batch size.
     *
     * Throws {@link IllegalArgumentException} if the batch size isn't positive.
     * @param batchSize The batch size to use.
     */
    public void setBatchSize(int batchSize) {
        if (batchSize > 0) {
            this.batchSize = batchSize;
        } else {
            throw new IllegalArgumentException("Batch size must be positive, found " + batchSize);
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
    protected TensorflowModel<T> copy(String newName, ModelProvenance newProvenance) {
        return new TensorflowModel<>(newName,newProvenance,featureIDMap,outputIDInfo,modelGraph.toGraphDef(),TensorflowUtil.serialise(modelGraph,session),batchSize,exampleTransformer,outputTransformer);
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
        Map<String,Object> tensorMap = TensorflowUtil.serialise(modelGraph, session);
        out.writeObject(tensorMap);
    }

    @SuppressWarnings("unchecked") //deserialising a typed map
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        byte[] modelBytes = (byte[]) in.readObject();
        Map<String,Object> tensorMap = (Map<String,Object>) in.readObject();
        modelGraph = new Graph();
        modelGraph.importGraphDef(modelBytes);
        session = new Session(modelGraph);
        // Initialises the parameters.
        session.runner().addTarget(TensorflowTrainer.INIT).run();
        TensorflowUtil.deserialise(session,tensorMap);
    }
}
