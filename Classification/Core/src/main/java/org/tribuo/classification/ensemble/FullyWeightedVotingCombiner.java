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

package org.tribuo.classification.ensemble;

import ai.onnx.proto.OnnxMl;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXUtils;

import java.util.ArrayList;
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
     * Constructs a weighted voting combiner.
     */
    public FullyWeightedVotingCombiner() {}

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
        Map<String,Label> predictionMap = new LinkedHashMap<>();
        for (int i = 0; i < score.length; i++) {
            String name = outputInfo.getOutput(i).getLabel();
            Label label = new Label(name,score[i]/sum);
            predictionMap.put(name,label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }

        Example<Label> example = predictions.get(0).getExample();

        return new Prediction<>(maxLabel,predictionMap,numUsed,example,true);
    }

    @Override
    public Prediction<Label> combine(ImmutableOutputInfo<Label> outputInfo, List<Prediction<Label>> predictions, float[] weights) {
        if (predictions.size() != weights.length) {
            throw new IllegalArgumentException("predictions and weights must be the same length. predictions.size()="+predictions.size()+", weights.length="+weights.length);
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
        Map<String,Label> predictionMap = new LinkedHashMap<>();
        for (int i = 0; i < score.length; i++) {
            String name = outputInfo.getOutput(i).getLabel();
            Label label = new Label(name,score[i]/sum);
            predictionMap.put(name,label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }

        Example<Label> example = predictions.get(0).getExample();

        return new Prediction<>(maxLabel,predictionMap,numUsed,example,true);
    }

    @Override
    public String toString() {
        return "FullyWeightedVotingCombiner()";
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

        // Take the mean over the maxed predictions
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

        // Multiply the input by the weights.
        OnnxMl.NodeProto mulByWeights = ONNXOperators.MUL.build(context,new String[]{input,unsqueeze.getOutput(0)},context.generateUniqueName("mul_predictions_by_weights"));
        nodes.add(mulByWeights);

        // Sum the weights
        OnnxMl.NodeProto weightSum = ONNXOperators.REDUCE_SUM.build(context,weight,context.generateUniqueName("ensemble_weight_sum"));
        nodes.add(weightSum);

        // Take the weighted mean over the outputs
        OnnxMl.TensorProto sumAxes = ONNXUtils.arrayBuilder(context,"sum_across_ensemble_axes",new long[]{2});
        context.addInitializer(sumAxes);
        Map<String,Object> attributes = new HashMap<>();
        attributes.put("keepdims",0);
        OnnxMl.NodeProto sumAcrossMembers = ONNXOperators.REDUCE_SUM.build(context,new String[]{mulByWeights.getOutput(0),sumAxes.getName()},context.generateUniqueName("sum_across_ensemble"),attributes);
        nodes.add(sumAcrossMembers);
        OnnxMl.NodeProto divideByWeightSum = ONNXOperators.DIV.build(context,new String[]{sumAcrossMembers.getOutput(0),weightSum.getOutput(0)},output);
        nodes.add(divideByWeightSum);

        return nodes;
    }
}
