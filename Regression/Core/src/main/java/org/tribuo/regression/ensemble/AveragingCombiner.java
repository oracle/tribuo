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

package org.tribuo.regression.ensemble;

import ai.onnx.proto.OnnxMl;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXUtils;
import org.tribuo.regression.Regressor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * A combiner which performs a weighted or unweighted average of the predicted
 * regressors independently across the output dimensions.
 */
public class AveragingCombiner implements EnsembleCombiner<Regressor> {
    private static final long serialVersionUID = 1L;

    /**
     * Constructs an averaging combiner.
     */
    public AveragingCombiner() {}

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

    /**
     * Exports this averaging combiner as a list of ONNX NodeProtos.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     * @param context The ONNX context object for name generation.
     * @param input The name of the input tensor to combine.
     * @param output The name of the averaged output.
     * @return A list of node protos representing the averaging operation.
     */
    @Override
    public List<OnnxMl.NodeProto> exportCombiner(ONNXContext context, String input, String output) {
        List<OnnxMl.NodeProto> nodes = new ArrayList<>();

        Map<String,Object> attributes = new HashMap<>();
        attributes.put("axes",new int[]{2});
        attributes.put("keepdims",0);
        OnnxMl.NodeProto mean = ONNXOperators.REDUCE_MEAN.build(context,input,output,attributes);
        nodes.add(mean);

        return nodes;
    }

    /**
     * Exports this ensemble combiner as a list of ONNX NodeProtos.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     * @param context The ONNX context object for name generation.
     * @param input The name of the input tensor to combine.
     * @param output The name of the averaged output.
     * @param weight The name of the combination weight initializer.
     * @return A list of node protos representing the averaging operation.
     */
    @Override
    public List<OnnxMl.NodeProto> exportCombiner(ONNXContext context, String input, String output, String weight) {
        List<OnnxMl.NodeProto> nodes = new ArrayList<>();

        // Unsqueeze the weights to make sure they broadcast how I want them too.
        // Now the size is [1, 1, num_members].
        OnnxMl.TensorProto unsqueezeAxes = ONNXUtils.arrayBuilder(context,"unsqueeze_ensemble_output",new long[]{0,1});
        context.addInitializer(unsqueezeAxes);
        OnnxMl.NodeProto unsqueeze = ONNXOperators.UNSQUEEZE.build(context,new String[]{weight,unsqueezeAxes.getName()},context.generateUniqueName("unsqueezed_weights"));
        nodes.add(unsqueeze);

        // Multiply the input by the weights.
        OnnxMl.NodeProto mulByWeights = ONNXOperators.MUL.build(context,new String[]{input,unsqueeze.getOutput(0)},context.generateUniqueName("mul_predictions_by_weights"));
        nodes.add(mulByWeights);

        // Sum the weights
        OnnxMl.NodeProto weightSum = ONNXOperators.REDUCE_SUM.build(context,weight,context.generateUniqueName("ensemble_weight_sum"));
        nodes.add(weightSum);

        // Take the mean
        OnnxMl.TensorProto sumAxes = ONNXUtils.arrayBuilder(context,"sum_across_ensemble_axes",new long[]{2});
        context.addInitializer(sumAxes);
        OnnxMl.NodeProto sumAcrossMembers = ONNXOperators.REDUCE_SUM.build(context,
                new String[]{mulByWeights.getOutput(0),sumAxes.getName()},
                context.generateUniqueName("sum_across_ensemble"),
                Collections.singletonMap("keepdims",0));
        nodes.add(sumAcrossMembers);
        OnnxMl.NodeProto divideByWeightSum = ONNXOperators.DIV.build(context,new String[]{sumAcrossMembers.getOutput(0),weightSum.getOutput(0)},output);
        nodes.add(divideByWeightSum);

        return nodes;
    }
}
