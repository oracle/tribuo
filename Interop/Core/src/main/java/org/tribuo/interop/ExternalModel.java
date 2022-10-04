/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop;

import org.tribuo.CategoricalInfo;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.MutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.Prediction;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * This is the base class for third party models which are trained externally and
 * loaded into Tribuo for prediction.
 * <p>
 * Batch size defaults to {@link ExternalModel#DEFAULT_BATCH_SIZE}
 * @param <T> The output subclass that this model operates on.
 * @param <U> The internal representation of features.
 * @param <V> The internal representation of outputs.
 */
public abstract class ExternalModel<T extends Output<T>,U,V> extends Model<T> {
    private static final long serialVersionUID = 1L;
    /**
     * Default batch size for external model batch predictions.
     */
    public static final int DEFAULT_BATCH_SIZE = 16;

    /**
     * The forward mapping from Tribuo's indices to the external indices.
     */
    protected final int[] featureForwardMapping;
    /**
     * The backward mapping from the external indices to Tribuo's indices.
     */
    protected final int[] featureBackwardMapping;

    private int batchSize = DEFAULT_BATCH_SIZE;

    /**
     * Constructs an external model from a model trained outside of Tribuo.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param generatesProbabilities Does this model generate probabilistic predictions.
     * @param featureMapping The mapping from Tribuo's feature names to the model's feature indices.
     */
    protected ExternalModel(String name, ModelProvenance provenance,
                            ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                            boolean generatesProbabilities, Map<String,Integer> featureMapping) {
        super(name, provenance, featureIDMap, outputIDInfo, generatesProbabilities);

        if (featureIDMap.size() != featureMapping.size()) {
            throw new IllegalArgumentException("The featureMapping must be the same size as the featureIDMap, found featureMapping.size()="+featureMapping.size()+", featureIDMap.size()="+featureIDMap.size());
        }

        this.featureForwardMapping = new int[featureIDMap.size()];
        this.featureBackwardMapping = new int[featureIDMap.size()];
        Arrays.fill(featureForwardMapping,-1);
        Arrays.fill(featureBackwardMapping,-1);

        for (Map.Entry<String,Integer> e : featureMapping.entrySet()) {
            int tribuoID = featureIDMap.getID(e.getKey());
            int mappingID = e.getValue();
            if (tribuoID == -1) {
                throw new IllegalArgumentException("Found invalid feature name in mapping " + e);
            } else if (mappingID >= featureForwardMapping.length) {
                throw new IllegalArgumentException("Found invalid feature id in mapping " + e);
            } else if (featureBackwardMapping[mappingID] != -1) {
                throw new IllegalArgumentException("Mapping for " + e + " already exists as feature " + featureIDMap.get(featureBackwardMapping[mappingID]));
            }

            featureForwardMapping[tribuoID] = mappingID;
            featureBackwardMapping[mappingID] = tribuoID;
        }
    }

    /**
     * Constructs an external model from a model trained outside of Tribuo.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param generatesProbabilities Does this model generate probabilistic predictions.
     * @param featureForwardMapping The mapping from Tribuo's indices to the model's indices.
     * @param featureBackwardMapping The mapping from the model's indices to Tribuo's indices.
     */
    protected ExternalModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, int[] featureForwardMapping, int[] featureBackwardMapping, boolean generatesProbabilities) {
        super(name,provenance,featureIDMap,outputIDInfo,generatesProbabilities);
        this.featureBackwardMapping = Arrays.copyOf(featureBackwardMapping,featureBackwardMapping.length);
        this.featureForwardMapping = Arrays.copyOf(featureForwardMapping,featureForwardMapping.length);
    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        SparseVector features = SparseVector.createSparseVector(example,featureIDMap,false);
        SparseVector renumberedFeatures = renumberFeatureIndices(features);
        U transformedFeatures = convertFeatures(renumberedFeatures);
        V output = externalPrediction(transformedFeatures);
        return convertOutput(output,features.numActiveElements(),example);
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

    private List<Prediction<T>> predictBatch(List<Example<T>> batch) {
        List<SparseVector> vectors = new ArrayList<>();
        int[] numValidFeatures = new int[batch.size()];
        for (int i = 0; i < batch.size(); i++) {
            SparseVector features = SparseVector.createSparseVector(batch.get(i),featureIDMap,false);
            vectors.add(renumberFeatureIndices(features));
            numValidFeatures[i] = features.numActiveElements();
        }
        U transformedFeatures = convertFeaturesList(vectors);
        V output = externalPrediction(transformedFeatures);
        List<Prediction<T>> predictions = convertOutput(output,numValidFeatures,batch);
        if (predictions.size() != vectors.size()) {
            throw new IllegalStateException("Unexpected number of predictions received from external model batch, found " + predictions.size() + ", expected " + vectors.size() + ".");
        } else {
            return predictions;
        }
    }

    /**
     * Renumbers the indices in a {@link SparseVector} switching from
     * Tribuo's internal indices to the external ones for this model.
     * @param input The features using internal indices.
     * @return The features using external indices.
     */
    private SparseVector renumberFeatureIndices(SparseVector input) {
        int inputSize = input.numActiveElements();
        int[] newIndices = new int[inputSize];
        double[] newValues = new double[inputSize];

        int i = 0;
        for (VectorTuple t : input) {
            int tribuoIdx = t.index;
            double value = t.value;
            newIndices[i] = featureForwardMapping[tribuoIdx];
            newValues[i] = value;
            i++;
        }

        return SparseVector.createSparseVector(input.size(),newIndices,newValues);
    }

    /**
     * Converts from a SparseVector using the external model's indices into
     * the ingestion format for the external model.
     * @param input The features using external indices.
     * @return The ingestion format for the external model.
     */
    protected abstract U convertFeatures(SparseVector input);

    /**
     * Converts from a list of SparseVector using the external model's indices
     * into the ingestion format for the external model.
     * @param input The features using external indices.
     * @return The ingestion format for the external model.
     */
    protected abstract U convertFeaturesList(List<SparseVector> input);

    /**
     * Runs the external model's prediction function.
     * @param input The input in the external model's format.
     * @return The output in the external model's format.
     */
    protected abstract V externalPrediction(U input);

    /**
     * Converts the output of the external model into a {@link Prediction}.
     * @param output The output of the external model.
     * @param numValidFeatures The number of valid features in the input.
     * @param example The input example, used to construct the Prediction.
     * @return A Tribuo Prediction.
     */
    protected abstract Prediction<T> convertOutput(V output, int numValidFeatures, Example<T> example);

    /**
     * Converts the output of the external model into a list of {@link Prediction}s.
     * @param output The output of the external model.
     * @param numValidFeatures An array with the number of valid features in each example.
     * @param examples The input examples, used to construct the Predictions.
     * @return A list of Tribuo Predictions.
     */
    protected abstract List<Prediction<T>> convertOutput(V output, int[] numValidFeatures, List<Example<T>> examples);

    /**
     * By default third party models don't return excuses.
     * @param example The input example.
     * @return Optional.empty.
     */
    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        return Optional.empty();
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
     * <p>
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
     * Creates an immutable feature map from a set of feature names.
     * <p>
     * Each feature is unobserved.
     * @param featureNames The names of the features to create.
     * @return A feature map representing the feature names.
     */
    protected static ImmutableFeatureMap createFeatureMap(Set<String> featureNames) {
        MutableFeatureMap featureMap = new MutableFeatureMap();

        for (String name : featureNames) {
            featureMap.put(new CategoricalInfo(name));
        }

        return new ImmutableFeatureMap(featureMap);
    }

    /**
     * Creates an output info from a set of outputs.
     * @param factory The output factory to use.
     * @param outputs The outputs and ids to observe.
     * @param <T> The type of the outputs.
     * @return An immutable output info representing the outputs.
     */
    protected static <T extends Output<T>> ImmutableOutputInfo<T> createOutputInfo(OutputFactory<T> factory, Map<T,Integer> outputs) {
        return factory.constructInfoForExternalModel(outputs);
    }

    /**
     * Checks if the feature mappings are valid for the supplied feature map.
     * @param featureForwardMapping The forward feature mapping.
     * @param featureBackwardMapping The backward feature mapping.
     * @param featureDomain The feature domain.
     * @return True if the feature mapping is valid (the forward and backward mappings are a bijection and the same size as the feature domain).
     */
    protected static boolean validateFeatureMapping(int[] featureForwardMapping, int[] featureBackwardMapping, ImmutableFeatureMap featureDomain) {
        if (featureBackwardMapping.length != featureForwardMapping.length) {
            return false;
        } else if (featureBackwardMapping.length != featureDomain.size()) {
            return false;
        } else {
            // check bijection
            Set<Integer> seenIndices = new HashSet<>();
            for (int tribuoId = 0; tribuoId < featureForwardMapping.length; tribuoId++) {
                int mappingId = featureForwardMapping[tribuoId];
                if (featureBackwardMapping[mappingId] != tribuoId) {
                    // not a bijection
                    return false;
                }
                seenIndices.add(mappingId);
            }
            // check for duplicate mapping
            return seenIndices.size() == featureDomain.size();
        }
    }
}
