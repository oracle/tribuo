/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.multilabel.ensemble;

import ai.onnx.proto.OnnxMl;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.math.la.DenseVector;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A combiner which performs a weighted or unweighted vote independently across the predicted labels in each multi-label.
 * <p>
 * This uses the thresholded predictions from each ensemble member.
 * <p>
 * This class is stateless and thread safe.
 */
public final class MultiLabelVotingCombiner implements EnsembleCombiner<MultiLabel> {
    private static final long serialVersionUID = 1L;

    /**
     * Constructs a voting combiner.
     */
    public MultiLabelVotingCombiner() {}

    @Override
    public Prediction<MultiLabel> combine(ImmutableOutputInfo<MultiLabel> outputInfo, List<Prediction<MultiLabel>> predictions) {
        int numPredictions = predictions.size();
        double weight = 1.0 / numPredictions;
        int numUsed = 0;
        double[] posScore = new double[outputInfo.size()];
        double[] negScore = new double[outputInfo.size()];
        for (Prediction<MultiLabel> p : predictions) {
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            DenseVector v = p.getOutput().convertToDenseVector(outputInfo);
            for (int j = 0; j < v.size(); j++) {
                double score = v.get(j);
                if (score > 0.5) {
                    posScore[j] += weight;
                } else {
                    negScore[j] += weight;
                }
            }
        }

        Map<String,MultiLabel> fullLabels = new LinkedHashMap<>();
        Set<Label> predSet = new HashSet<>();
        for (int i = 0; i < posScore.length; i++) {
            String name = outputInfo.getOutput(i).getLabelString();
            double score = posScore[i] / (posScore[i] + negScore[i]);
            Label label = new Label(name, score);
            if (score > 0.5) {
                predSet.add(label);
            }
            fullLabels.put(name,new MultiLabel(label));
        }

        Example<MultiLabel> example = predictions.get(0).getExample();

        return new Prediction<>(new MultiLabel(predSet),fullLabels,numUsed,example,true);
    }

    @Override
    public Prediction<MultiLabel> combine(ImmutableOutputInfo<MultiLabel> outputInfo, List<Prediction<MultiLabel>> predictions, float[] weights) {
        if (predictions.size() != weights.length) {
            throw new IllegalArgumentException("predictions and weights must be the same length. predictions.size()="+predictions.size()+", weights.length="+weights.length);
        }
        int numUsed = 0;
        double[] posScore = new double[outputInfo.size()];
        double[] negScore = new double[outputInfo.size()];
        for (int i = 0; i < weights.length; i++) {
            Prediction<MultiLabel> p = predictions.get(i);
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            DenseVector v = p.getOutput().convertToDenseVector(outputInfo);
            for (int j = 0; j < v.size(); j++) {
               double score = v.get(j);
               if (score > 0.5) {
                   posScore[j] += weights[i];
               } else {
                   negScore[j] += weights[i];
               }
            }
        }

        Map<String,MultiLabel> fullLabels = new LinkedHashMap<>();
        Set<Label> predSet = new HashSet<>();
        for (int i = 0; i < posScore.length; i++) {
            String name = outputInfo.getOutput(i).getLabelString();
            double score = posScore[i] / (posScore[i] + negScore[i]);
            Label label = new Label(name, score);
            if (score > 0.5) {
                predSet.add(label);
            }
            fullLabels.put(name,new MultiLabel(label));
        }

        Example<MultiLabel> example = predictions.get(0).getExample();

        return new Prediction<>(new MultiLabel(predSet),fullLabels,numUsed,example,true);
    }

    @Override
    public String toString() {
        return "MultiLabelVotingCombiner()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"EnsembleCombiner");
    }

    /**
     * Exports this voting combiner as a list of ONNX NodeProtos.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     * @param context The ONNX context object for name generation.
     * @param input The name of the input tensor to combine.
     * @param output The name of the voting output.
     * @return A list of node protos representing the voting operation.
     */
    @Override
    public List<OnnxMl.NodeProto> exportCombiner(ONNXContext context, String input, String output) {
        List<OnnxMl.NodeProto> nodes = new ArrayList<>();

        // greater than 0.5
        OnnxMl.TensorProto half = ONNXUtils.scalarBuilder(context,"half",0.5f);
        context.addInitializer(half);
        OnnxMl.NodeProto greater = ONNXOperators.GREATER.build(context, new String[]{input,half.getName()},context.generateUniqueName("combiner_greater"));
        nodes.add(greater);

        // where 1 v 0
        OnnxMl.TensorProto one = ONNXUtils.scalarBuilder(context,"one",1.0f);
        context.addInitializer(one);
        OnnxMl.TensorProto zero = ONNXUtils.scalarBuilder(context,"zero",0.0f);
        context.addInitializer(zero);
        OnnxMl.NodeProto where = ONNXOperators.WHERE.build(context,
                new String[]{greater.getOutput(0),one.getName(),zero.getName()},
                context.generateUniqueName("combiner_where"));
        nodes.add(where);

        // Take the mean over the where'd predictions
        Map<String,Object> attributes = new HashMap<>();
        attributes.put("axes",new int[]{2});
        attributes.put("keepdims",0);
        OnnxMl.NodeProto mean = ONNXOperators.REDUCE_MEAN.build(context,where.getOutput(0),output,attributes);
        nodes.add(mean);

        return nodes;
    }

    /**
     * Exports this voting combiner as a list of ONNX NodeProtos.
     * <p>
     * The input should be a 3-tensor [batch_size, num_outputs, num_ensemble_members].
     * @param context The ONNX context object for name generation.
     * @param input The name of the input tensor to combine.
     * @param output The name of the voting output.
     * @param weight The name of the combination weight initializer.
     * @return A list of node protos representing the voting operation.
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

        // greater than 0.5
        OnnxMl.TensorProto half = ONNXUtils.scalarBuilder(context,"half",0.5f);
        context.addInitializer(half);
        OnnxMl.NodeProto greater = ONNXOperators.GREATER.build(context, new String[]{input,half.getName()},context.generateUniqueName("combiner_greater"));
        nodes.add(greater);

        // where 1 v 0
        OnnxMl.TensorProto one = ONNXUtils.scalarBuilder(context,"one",1.0f);
        context.addInitializer(one);
        OnnxMl.TensorProto zero = ONNXUtils.scalarBuilder(context,"zero",0.0f);
        context.addInitializer(zero);
        OnnxMl.NodeProto where = ONNXOperators.WHERE.build(context,
                new String[]{greater.getOutput(0),one.getName(),zero.getName()},
                context.generateUniqueName("combiner_where"));
        nodes.add(where);

        // Multiply the where'd input by the weights.
        OnnxMl.NodeProto mulByWeights = ONNXOperators.MUL.build(context,new String[]{where.getOutput(0),unsqueeze.getOutput(0)},context.generateUniqueName("mul_predictions_by_weights"));
        nodes.add(mulByWeights);

        // Sum the weights
        OnnxMl.NodeProto weightSum = ONNXOperators.REDUCE_SUM.build(context,weight,context.generateUniqueName("ensemble_weight_sum"));
        nodes.add(weightSum);

        // Take the weighted mean over the outputs
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
