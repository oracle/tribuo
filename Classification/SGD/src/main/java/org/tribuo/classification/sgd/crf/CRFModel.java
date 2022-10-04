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

package org.tribuo.classification.sgd.crf;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.sequence.ConfidencePredictingSequenceModel;
import org.tribuo.classification.sgd.protos.CRFModelProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.protos.core.SequenceModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.sequence.SequenceExample;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.logging.Logger;

import static org.tribuo.Model.BIAS_FEATURE;

/**
 * An inference time model for a linear chain CRF trained using SGD.
 * <p>
 * Can be switched to use Viterbi, belief propagation, or constrained BP at test time. By default it uses Viterbi.
 * <p>
 * See:
 * <pre>
 * Lafferty J, McCallum A, Pereira FC.
 * "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"
 * Proceedings of the 18th International Conference on Machine Learning 2001 (ICML 2001).
 * </pre>
 */
public class CRFModel extends ConfidencePredictingSequenceModel {
    private static final Logger logger = Logger.getLogger(CRFModel.class.getName());
    private static final long serialVersionUID = 2L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final CRFParameters parameters;

    /**
     * The type of subsequence level confidence to predict.
     */
    public enum ConfidenceType {
        /**
         * No confidence predction.
         */
        NONE,
        /**
         * Belief Propagation
         */
        MULTIPLY,
        /**
         * Constrained Belief Propagation from "Confidence Estimation for Information Extraction" Culotta and McCallum 2004.
         */
        CONSTRAINED_BP
    }

    private ConfidenceType confidenceType;

    CRFModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap, CRFParameters parameters) {
        super(name, description, featureIDMap, labelIDMap);
        this.parameters = parameters;
        this.confidenceType = ConfidenceType.NONE;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static CRFModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        CRFModelProto proto = message.unpack(CRFModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Label.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a label domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Label> outputDomain = (ImmutableOutputInfo<Label>) carrier.outputDomain();

        Parameters params = Parameters.deserialize(proto.getParams());
        if (!(params instanceof CRFParameters)) {
            throw new IllegalStateException("Invalid protobuf, parameters must be CRFParameters, found " + params.getClass());
        }

        ConfidenceType confidenceType = ConfidenceType.valueOf(proto.getConfidenceType());

        CRFModel model = new CRFModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),outputDomain,(CRFParameters) params);
        model.confidenceType = confidenceType;
        return model;
    }

    /**
     * Sets the inference method used for confidence prediction.
     * If CONSTRAINED_BP uses the constrained belief propagation algorithm from Culotta and McCallum 2004,
     * if MULTIPLY multiplies the maximum marginal for each token, if NONE uses Viterbi.
     *
     * @param type Enum specifying the confidence type.
     */
    public void setConfidenceType(ConfidenceType type) {
        this.confidenceType = type;
    }

    /**
     * Get a copy of the weights for feature {@code featureID}.
     * @param featureID The feature ID.
     * @return The per class weights.
     */
    public DenseVector getFeatureWeights(int featureID) {
        if (featureID < 0 || featureID >= featureIDMap.size()) {
            logger.warning("Unknown feature");
            return new DenseVector(0);
        } else {
            return parameters.getFeatureWeights(featureID);
        }
    }

    /**
     * Get a copy of the weights for feature named {@code featureName}.
     * @param featureName The feature name.
     * @return The per class weights.
     */
    public DenseVector getFeatureWeights(String featureName) {
        int id = featureIDMap.getID(featureName);
        if (id > -1) {
            return getFeatureWeights(featureIDMap.getID(featureName));
        } else {
            logger.warning("Unknown feature");
            return new DenseVector(0);
        }
    }

    @Override
    public List<Prediction<Label>> predict(SequenceExample<Label> example) {
        SGDVector[] features = convertToVector(example,featureIDMap);
        List<Prediction<Label>> output = new ArrayList<>();
        if (confidenceType == ConfidenceType.MULTIPLY) {
            DenseVector[] marginals = parameters.predictMarginals(features);

            for (int i = 0; i < marginals.length; i++) {
                double maxScore = Double.NEGATIVE_INFINITY;
                Label maxLabel = null;
                Map<String,Label> predMap = new LinkedHashMap<>();
                for (int j = 0; j < marginals[i].size(); j++) {
                    String labelName = outputIDMap.getOutput(j).getLabel();
                    Label label = new Label(labelName, marginals[i].get(j));
                    predMap.put(labelName, label);
                    if (label.getScore() > maxScore) {
                        maxScore = label.getScore();
                        maxLabel = label;
                    }
                }
                output.add(new Prediction<>(maxLabel, predMap, features[i].numActiveElements(), example.get(i), true));
            }
        } else {
            int[] predLabels = parameters.predict(features);

            for (int i = 0; i < predLabels.length; i++) {
                output.add(new Prediction<>(outputIDMap.getOutput(predLabels[i]),features[i].numActiveElements(),example.get(i)));
            }
        }

        return output;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        int maxFeatures = n < 0 ? featureIDMap.size() + 1 : n;

        // Uses a standard comparator rather than a Math.abs comparator as it's a log-linear model
        // so nothing is actually negative.
        Comparator<Pair<String,Double>> comparator = Comparator.comparing(Pair::getB);

        //
        // Use a priority queue to find the top N features.
        int numClasses = outputIDMap.size();
        int numFeatures = featureIDMap.size();
        Map<String, List<Pair<String,Double>>> map = new HashMap<>();
        for (int i = 0; i < numClasses; i++) {
            PriorityQueue<Pair<String,Double>> q = new PriorityQueue<>(maxFeatures, comparator);

            for (int j = 0; j < numFeatures; j++) {
                Pair<String,Double> curr = new Pair<>(featureIDMap.get(j).getName(), parameters.getWeight(i,j));

                if (q.size() < maxFeatures) {
                    q.offer(curr);
                } else if (comparator.compare(curr, q.peek()) > 0) {
                    q.poll();
                    q.offer(curr);
                }
            }
            Pair<String,Double> curr = new Pair<>(BIAS_FEATURE, parameters.getBias(i));

            if (q.size() < maxFeatures) {
                q.offer(curr);
            } else if (comparator.compare(curr, q.peek()) > 0) {
                q.poll();
                q.offer(curr);
            }
            List<Pair<String,Double>> b = new ArrayList<>();
            while (q.size() > 0) {
                b.add(q.poll());
            }

            Collections.reverse(b);
            map.put(outputIDMap.getOutput(i).getLabel(), b);
        }
        return map;
    }

    @Override
    public <SUB extends Subsequence> List<Double> scoreSubsequences(SequenceExample<Label> example, List<Prediction<Label>> predictions, List<SUB> subsequences) {
        if (confidenceType == ConfidenceType.CONSTRAINED_BP) {
            List<Chunk> chunks = new ArrayList<>();
            for(Subsequence subsequence : subsequences) {
                int[] ids = new int[subsequence.length()];
                for(int i=0; i<ids.length; i++) {
                    ids[i] = outputIDMap.getID(predictions.get(i+subsequence.begin).getOutput());
                }
                chunks.add(new Chunk(subsequence.begin, ids));
            }
            return scoreChunks(example, chunks);
        } else {
            return ConfidencePredictingSequenceModel.multiplyWeights(predictions, subsequences);
        }
    }

    /**
     * Scores the chunks using constrained belief propagation.
     * @param example The example to score.
     * @param chunks The predicted chunks.
     * @return The scores.
     */
    public List<Double> scoreChunks(SequenceExample<Label> example, List<Chunk> chunks) {
        SGDVector[] features = convertToVector(example,featureIDMap);
        return parameters.predictConfidenceUsingCBP(features,chunks);
    }

    /**
     * Generates a human readable string containing all the weights in this model.
     * @return A string containing all the weight values.
     */
    public String generateWeightsString() {
        StringBuilder buffer = new StringBuilder();

        Tensor[] weights = parameters.get();

        buffer.append("Biases = ");
        buffer.append(weights[0].toString());
        buffer.append('\n');

        buffer.append("Feature-Label weights = \n");
        buffer.append(weights[1].toString());
        buffer.append('\n');

        buffer.append("Label-Label weights = \n");
        buffer.append(weights[2].toString());
        buffer.append('\n');

        return buffer.toString();
    }

    @Override
    public SequenceModelProto serialize() {
        ModelDataCarrier<Label> carrier = createDataCarrier();
        CRFModelProto.Builder modelBuilder = CRFModelProto.newBuilder();
        modelBuilder.setConfidenceType(confidenceType.name());
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setParams(parameters.serialize());

        SequenceModelProto.Builder builder = SequenceModelProto.newBuilder();
        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(CRFModel.class.getName());
        builder.setSerializedData(Any.pack(modelBuilder.build()));

        return builder.build();
    }

    /**
     * Converts a {@link SequenceExample} into an array of {@link SparseVector}s suitable for CRF prediction.
     * @deprecated As it's replaced with {@link #convertToVector} which is more flexible.
     * @param example The sequence example to convert
     * @param featureIDMap The feature id map, used to discover the number of features.
     * @param <T> The type parameter of the sequence example.
     * @return An array of {@link SparseVector}.
     */
    @Deprecated
    public static <T extends Output<T>> SparseVector[] convert(SequenceExample<T> example, ImmutableFeatureMap featureIDMap) {
        int length = example.size();
        if (length == 0) {
            throw new IllegalArgumentException("SequenceExample is empty, " + example.toString());
        }
        SparseVector[] features = new SparseVector[length];
        int i = 0;
        for (Example<T> e : example) {
            features[i] = SparseVector.createSparseVector(e,featureIDMap,false);
            if (features[i].numActiveElements() == 0) {
                throw new IllegalArgumentException("No features found in Example " + e.toString());
            }
            i++;
        }
        return features;
    }

    /**
     * Converts a {@link SequenceExample} into an array of {@link SparseVector}s and labels suitable for CRF prediction.
     * @deprecated As it's replaced with {@link #convertToVector} which is more flexible.
     * @param example The sequence example to convert
     * @param featureIDMap The feature id map, used to discover the number of features.
     * @param labelIDMap The label id map, used to get the index of the labels.
     * @return A {@link Pair} of an int array of labels and an array of {@link SparseVector}.
     */
    @Deprecated
    public static Pair<int[],SparseVector[]> convert(SequenceExample<Label> example, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap) {
        int length = example.size();
        if (length == 0) {
            throw new IllegalArgumentException("SequenceExample is empty, " + example.toString());
        }
        int[] labels = new int[length];
        SparseVector[] features = new SparseVector[length];
        int i = 0;
        for (Example<Label> e : example) {
            labels[i] = labelIDMap.getID(e.getOutput());
            features[i] = SparseVector.createSparseVector(e,featureIDMap,false);
            if (features[i].numActiveElements() == 0) {
                throw new IllegalArgumentException("No features found in Example " + e.toString());
            }
            i++;
        }
        return new Pair<>(labels,features);
    }

    /**
     * Converts a {@link SequenceExample} into an array of {@link SGDVector}s suitable for CRF prediction.
     * @param example The sequence example to convert
     * @param featureIDMap The feature id map, used to discover the number of features.
     * @param <T> The type parameter of the sequence example.
     * @return An array of {@link SGDVector}.
     */
    public static <T extends Output<T>> SGDVector[] convertToVector(SequenceExample<T> example, ImmutableFeatureMap featureIDMap) {
        int length = example.size();
        if (length == 0) {
            throw new IllegalArgumentException("SequenceExample is empty, " + example.toString());
        }
        int featureSpaceSize = featureIDMap.size();
        SGDVector[] features = new SGDVector[length];
        int i = 0;
        for (Example<T> e : example) {
            if (e.size() == featureSpaceSize) {
                features[i] = DenseVector.createDenseVector(e, featureIDMap, false);
            } else {
                features[i] = SparseVector.createSparseVector(e, featureIDMap, false);
            }
            if (features[i].numActiveElements() == 0) {
                throw new IllegalArgumentException("No features found in Example " + e.toString());
            }
            i++;
        }
        return features;
    }

    /**
     * Converts a {@link SequenceExample} into an array of {@link SGDVector}s and labels suitable for CRF prediction.
     * @param example The sequence example to convert
     * @param featureIDMap The feature id map, used to discover the number of features.
     * @param labelIDMap The label id map, used to get the index of the labels.
     * @return A {@link Pair} of an int array of labels and an array of {@link SparseVector}.
     */
    public static Pair<int[],SGDVector[]> convertToVector(SequenceExample<Label> example, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap) {
        int length = example.size();
        if (length == 0) {
            throw new IllegalArgumentException("SequenceExample is empty, " + example.toString());
        }
        int featureSpaceSize = featureIDMap.size();
        int[] labels = new int[length];
        SGDVector[] features = new SGDVector[length];
        int i = 0;
        for (Example<Label> e : example) {
            labels[i] = labelIDMap.getID(e.getOutput());
            if (e.size() == featureSpaceSize) {
                features[i] = DenseVector.createDenseVector(e, featureIDMap, false);
            } else {
                features[i] = SparseVector.createSparseVector(e, featureIDMap, false);
            }
            if (features[i].numActiveElements() == 0) {
                throw new IllegalArgumentException("No features found in Example " + e.toString());
            }
            i++;
        }
        return new Pair<>(labels,features);
    }
}
