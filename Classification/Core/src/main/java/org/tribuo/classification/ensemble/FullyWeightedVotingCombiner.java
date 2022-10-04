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

package org.tribuo.classification.ensemble;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.protos.core.EnsembleCombinerProto;
import org.tribuo.util.onnx.ONNXInitializer;
import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXOperators;
import org.tribuo.util.onnx.ONNXRef;

import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A combiner which performs a weighted or unweighted vote across the predicted labels.
 * <p>
 * This uses the full distribution of predictions from each ensemble member, unlike {@link VotingCombiner}
 * which uses the most likely prediction for each ensemble member.
 */
public final class FullyWeightedVotingCombiner implements EnsembleCombiner<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs a weighted voting combiner.
     */
    public FullyWeightedVotingCombiner() {}

    /**
     * Deserialization factory.
     *
     * @param version   The serialized object version.
     * @param className The class name.
     * @param message   The serialized data.
     * @return The deserialized object.
     */
    public static FullyWeightedVotingCombiner deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new FullyWeightedVotingCombiner();
    }

    @Override
    public EnsembleCombinerProto serialize() {
        EnsembleCombinerProto.Builder combinerProto = EnsembleCombinerProto.newBuilder();
        combinerProto.setClassName(this.getClass().getName());
        combinerProto.setVersion(CURRENT_VERSION);
        return combinerProto.build();
    }

    @Override
    public Prediction<Label> combine(ImmutableOutputInfo<Label> outputInfo, List<Prediction<Label>> predictions) {
        int numPredictions = predictions.size();
        int numUsed = 0;
        double weight = 1.0 / numPredictions;
        double sum = 0.0;
        double[] score = new double[outputInfo.size()];
        for (Prediction<Label> p : predictions) {
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            for (Label e : p.getOutputScores().values()) {
                double curScore = weight * e.getScore();
                sum += curScore;
                score[outputInfo.getID(e)] += curScore;
            }
        }

        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String, Label> predictionMap = new LinkedHashMap<>();
        for (int i = 0; i < score.length; i++) {
            String name = outputInfo.getOutput(i).getLabel();
            Label label = new Label(name, score[i] / sum);
            predictionMap.put(name, label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }

        Example<Label> example = predictions.get(0).getExample();

        return new Prediction<>(maxLabel, predictionMap, numUsed, example, true);
    }

    @Override
    public Prediction<Label> combine(ImmutableOutputInfo<Label> outputInfo, List<Prediction<Label>> predictions, float[] weights) {
        if (predictions.size() != weights.length) {
            throw new IllegalArgumentException("predictions and weights must be the same length. predictions.size()=" + predictions.size() + ", weights.length=" + weights.length);
        }
        int numUsed = 0;
        double sum = 0.0;
        double[] score = new double[outputInfo.size()];
        for (int i = 0; i < weights.length; i++) {
            Prediction<Label> p = predictions.get(i);
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            for (Label e : p.getOutputScores().values()) {
                double curScore = weights[i] * e.getScore();
                sum += curScore;
                score[outputInfo.getID(e)] += curScore;
            }
        }

        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String, Label> predictionMap = new LinkedHashMap<>();
        for (int i = 0; i < score.length; i++) {
            String name = outputInfo.getOutput(i).getLabel();
            Label label = new Label(name, score[i] / sum);
            predictionMap.put(name, label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }

        Example<Label> example = predictions.get(0).getExample();

        return new Prediction<>(maxLabel, predictionMap, numUsed, example, true);
    }

    @Override
    public String toString() {
        return "FullyWeightedVotingCombiner()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "EnsembleCombiner");
    }

    @Override
    public Class<Label> getTypeWitness() {
        return Label.class;
    }

    /**
     * Exports this voting combiner to ONNX.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     *
     * @param input the node to be ensembled according to this implementation.
     * @return The leaf node of the voting operation.
     */
    @Override
    public ONNXNode exportCombiner(ONNXNode input) {
        // Take the mean over the maxed predictions
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("axes", new int[]{2});
        attributes.put("keepdims", 0);
        return input.apply(ONNXOperators.REDUCE_MEAN, attributes);
    }

    /**
     * Exports this voting combiner to ONNX.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     *
     * @param input  the node to be ensembled according to this implementation.
     * @param weight The node of weights for ensembling.
     * @return The leaf node of the voting operation.
     */
    @Override
    public <T extends ONNXRef<?>> ONNXNode exportCombiner(ONNXNode input, T weight) {
        // Unsqueeze the weights to make sure they broadcast how I want them too.
        // Now the size is [1, 1, num_members].
        ONNXInitializer unsqueezeAxes = input.onnxContext().array("unsqueeze_ensemble_output", new long[]{0, 1});

        ONNXNode unsqueezed = weight.apply(ONNXOperators.UNSQUEEZE, unsqueezeAxes);

        // Multiply the input by the weights.
        ONNXNode mulByWeights = input.apply(ONNXOperators.MUL, unsqueezed);

        // Sum the weights
        ONNXNode weightSum = weight.apply(ONNXOperators.REDUCE_SUM);

        // Take the weighted mean over the outputs
        ONNXInitializer sumAxes = input.onnxContext().array("sum_across_ensemble_axes", new long[]{2});
        return mulByWeights.apply(ONNXOperators.REDUCE_SUM, sumAxes, Collections.singletonMap("keepdims", 0))
                .apply(ONNXOperators.DIV, weightSum);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof FullyWeightedVotingCombiner;
    }

    @Override
    public int hashCode() {
        return 31;
    }
}
