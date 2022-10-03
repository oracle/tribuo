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

package org.tribuo.regression.ensemble;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.protos.core.EnsembleCombinerProto;
import org.tribuo.regression.Regressor;
import org.tribuo.util.onnx.ONNXContext;
import org.tribuo.util.onnx.ONNXInitializer;
import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXOperators;
import org.tribuo.util.onnx.ONNXRef;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A combiner which performs a weighted or unweighted average of the predicted
 * regressors independently across the output dimensions.
 */
public class AveragingCombiner implements EnsembleCombiner<Regressor> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs an averaging combiner.
     */
    public AveragingCombiner() {}

    /**
     * Deserialization factory.
     *
     * @param version   The serialized object version.
     * @param className The class name.
     * @param message   The serialized data.
     * @return The deserialized object.
     */
    public static AveragingCombiner deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new AveragingCombiner();
    }

    @Override
    public EnsembleCombinerProto serialize() {
        EnsembleCombinerProto.Builder combinerProto = EnsembleCombinerProto.newBuilder();
        combinerProto.setClassName(this.getClass().getName());
        combinerProto.setVersion(CURRENT_VERSION);
        return combinerProto.build();
    }

    @Override
    public Prediction<Regressor> combine(ImmutableOutputInfo<Regressor> outputInfo, List<Prediction<Regressor>> predictions) {
        int numPredictions = predictions.size();
        int dimensions = outputInfo.size();
        int numUsed = 0;
        String[] names;
        double[] mean = new double[dimensions];
        double[] variance = new double[dimensions];
        for (Prediction<Regressor> p : predictions) {
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            Regressor curValue = p.getOutput();
            for (int i = 0; i < dimensions; i++) {
                double value = curValue.getValues()[i];
                double oldMean = mean[i];
                mean[i] += (value - oldMean);
                variance[i] += (value - oldMean) * (value - mean[i]);
            }
        }
        names = predictions.get(0).getOutput().getNames();
        if (numPredictions > 1) {
            for (int i = 0; i < dimensions; i++) {
                variance[i] /= (numPredictions-1);
            }
        } else {
            Arrays.fill(variance,0);
        }

        Example<Regressor> example = predictions.get(0).getExample();
        return new Prediction<>(new Regressor(names,mean,variance),numUsed,example);
    }

    @Override
    public Prediction<Regressor> combine(ImmutableOutputInfo<Regressor> outputInfo, List<Prediction<Regressor>> predictions, float[] weights) {
        if (predictions.size() != weights.length) {
            throw new IllegalArgumentException("predictions and weights must be the same length. predictions.size()="+predictions.size()+", weights.length="+weights.length);
        }
        int dimensions = outputInfo.size();
        int numUsed = 0;
        String[] names;
        double[] mean = new double[dimensions];
        double[] variance = new double[dimensions];
        double weightSum = 0.0;
        for (int i = 0; i < weights.length; i++) {
            Prediction<Regressor> p = predictions.get(i);
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            Regressor curValue = p.getOutput();
            float weight = weights[i];
            weightSum += weight;
            for (int j = 0; j < dimensions; j++) {
                double value = curValue.getValues()[j];
                double oldMean = mean[j];
                mean[j] += (weight / weightSum) * (value - oldMean);
                variance[j] += weight * (value - oldMean) * (value - mean[j]);
            }
        }
        names = predictions.get(0).getOutput().getNames();
        if (weights.length > 1) {
            for (int i = 0; i < dimensions; i++) {
                variance[i] /= (weightSum-1);
            }
        } else {
            Arrays.fill(variance,0);
        }

        Example<Regressor> example = predictions.get(0).getExample();
        return new Prediction<>(new Regressor(names,mean,variance),numUsed,example);
    }

    @Override
    public String toString() {
        return "MultipleOutputAveragingCombiner()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"EnsembleCombiner");
    }

    @Override
    public Class<Regressor> getTypeWitness() {
        return Regressor.class;
    }

    /**
     * Exports this averaging combiner, writing constructed nodes into the {@link ONNXContext}
     * governing {@code input} and returning the leaf node of the combiner.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     * @param input The node to combine
     * @return A node representing the final average operation.
     */
    @Override
    public ONNXNode exportCombiner(ONNXNode input) {
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("axes", new int[]{2});
        attributes.put("keepdims", 0);
        return input.apply(ONNXOperators.REDUCE_MEAN, attributes);
    }

    /**
     * Exports this averaging combiner, writing constructed nodes into the {@link ONNXContext}
     * governing {@code input} and returning the leaf node of the combiner.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     * @param input The node to combine
     * @param weight The node of weights to use in combining.
     * @return A node representing the final average operation.
     */
    @Override
    public <T extends ONNXRef<?>> ONNXNode exportCombiner(ONNXNode input, T weight) {
        // Unsqueeze the weights to make sure they broadcast how I want them too.
        // Now the size is [1, 1, num_members].
        ONNXInitializer unsqueezeAxes = input.onnxContext().array("unsqueeze_ensemble_output", new long[]{0, 1});
        ONNXInitializer sumAxes = input.onnxContext().array("sum_across_ensemble_axes", new long[]{2});

        ONNXNode unsqueezed = weight.apply(ONNXOperators.UNSQUEEZE, unsqueezeAxes);

        // Multiply the input by the weights.
        ONNXNode mulByWeights = input.apply(ONNXOperators.MUL, unsqueezed);

        // Sum the weights
        ONNXNode weightSum = weight.apply(ONNXOperators.REDUCE_SUM);


        // Take the mean
        return mulByWeights.apply(ONNXOperators.REDUCE_SUM, sumAxes, Collections.singletonMap("keepdims", 0))
                .apply(ONNXOperators.DIV, weightSum);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof AveragingCombiner;
    }

    @Override
    public int hashCode() {
        return 31;
    }
}
