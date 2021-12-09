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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.SessionFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.proto.framework.GraphDef;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Base class for a TensorFlow model that operates on {@link Example}s.
 * <p>
 * The subclasses are package private and concern themselves with how the model is stored
 * on disk.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 * @param <T> The output type.
 */
public abstract class TensorFlowModel<T extends Output<T>> extends Model<T> implements AutoCloseable {
    private static final Logger logger = Logger.getLogger(TensorFlowModel.class.getName());
    private static final long serialVersionUID = 200L;

    protected int batchSize;
    protected final String outputName;
    protected final FeatureConverter featureConverter;
    protected final OutputConverter<T> outputConverter;
    protected transient Graph modelGraph = null;
    protected transient Session session = null;
    protected transient boolean closed = false;

    /**
     * Builds a TFModel. The session should be initialized in the subclass constructor.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param trainedGraphDef The graph definition.
     * @param batchSize The test time batch size.
     * @param outputName The name of the output operation.
     * @param featureConverter The feature converter.
     * @param outputConverter The output converter.
     */
    protected TensorFlowModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, GraphDef trainedGraphDef, int batchSize, String outputName, FeatureConverter featureConverter, OutputConverter<T> outputConverter) {
        super(name, provenance, featureIDMap, outputIDInfo, outputConverter.generatesProbabilities());
        this.modelGraph = new Graph();
        this.modelGraph.importGraphDef(trainedGraphDef);
        this.session = new Session(modelGraph);
        this.batchSize = batchSize;
        this.outputName = outputName;
        this.featureConverter = featureConverter;
        this.outputConverter = outputConverter;
    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        if (closed) {
            throw new IllegalStateException("Can't use a closed model, the state has gone.");
        }
        // This adds overhead and triggers lookups for each feature, but is necessary to correctly calculate
        // the number of features used in this example.
        SparseVector vec = SparseVector.createSparseVector(example, featureIDMap, false);
        try (TensorMap transformedInput = featureConverter.convert(vec);
             Tensor outputTensor = transformedInput.feedInto(session.runner())
                     .fetch(outputName).run().get(0)) {
            // Transform the returned tensor into a Prediction.
            return outputConverter.convertToPrediction(outputTensor, outputIDInfo, vec.numActiveElements(), example);
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
        if (closed) {
            throw new IllegalStateException("Can't use a closed model, the state has gone.");
        }
        // Convert the batch
        List<SparseVector> vectors = new ArrayList<>(batchExamples.size());
        int[] numActiveElements = new int[batchExamples.size()];
        for (int i = 0; i < batchExamples.size(); i++) {
            SparseVector vec = SparseVector.createSparseVector(batchExamples.get(i), featureIDMap, false);
            numActiveElements[i] = vec.numActiveElements();
            vectors.add(vec);
        }

        // Send a batch to Tensorflow
        try (TensorMap transformedInput = featureConverter.convert(vectors);
             Tensor outputTensor = transformedInput.feedInto(session.runner())
                     .fetch(outputName).run().get(0)) {
            // Transform the returned tensor into a list of Predictions.
            return outputConverter.convertToBatchPrediction(outputTensor, outputIDInfo, numActiveElements, batchExamples);
        }
    }

    /**
     * Gets the current testing batch size.
     *
     * @return The batch size.
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Sets a new batch size.
     * <p>
     * Throws {@link IllegalArgumentException} if the batch size isn't positive.
     *
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
     *
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
     *
     * @param example The input example.
     * @return {@link Optional#empty}.
     */
    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        return Optional.empty();
    }

    /**
     * Gets the name of the output operation.
     * @return The output operation name.
     */
    public String getOutputName() {
        return outputName;
    }

    /**
     * Exports this model as a {@link SavedModelBundle}, writing to the supplied directory.
     *
     * @param path The directory to export to.
     * @throws IOException If it failed to write to the directory.
     */
    public void exportModel(String path) throws IOException {
        if (closed) {
            throw new IllegalStateException("Can't serialize a closed model, the state has gone.");
        }
        Signature.Builder sigBuilder = Signature.builder();
        Set<String> inputs = featureConverter.inputNamesSet();
        for (String s : inputs) {
            Operation inputOp = modelGraph.operation(s);
            sigBuilder.input(s, inputOp.output(0));
        }
        Operation outputOp = modelGraph.operation(outputName);
        Signature modelSig = sigBuilder.output(outputName, outputOp.output(0)).build();
        SessionFunction concFunc = SessionFunction.create(modelSig, session);
        SavedModelBundle.exporter(path).withFunction(concFunc).export();
    }

    @Override
    public void close() {
        if (session != null) {
            session.close();
            session = null;
        }
        if (modelGraph != null) {
            modelGraph.close();
            modelGraph = null;
        }
        closed = true;
    }

}
