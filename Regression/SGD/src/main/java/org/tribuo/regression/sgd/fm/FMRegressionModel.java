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

package org.tribuo.regression.sgd.fm;

import ai.onnx.proto.OnnxMl;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.common.sgd.AbstractFMModel;
import org.tribuo.common.sgd.FMParameters;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXShape;
import org.tribuo.math.onnx.ONNXMathUtils;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.ImmutableRegressionInfo;
import org.tribuo.regression.Regressor;

import java.util.Arrays;

/**
 * The inference time model of a regression factorization machine trained using SGD.
 * Independently predicts each output dimension, unless they are tied together in the
 * optimiser.
 * <p>
 * See:
 * <pre>
 * Rendle, S.
 * Factorization machines.
 * 2010 IEEE International Conference on Data Mining
 * </pre>
 */
public class FMRegressionModel extends AbstractFMModel<Regressor> implements ONNXExportable {
    private static final long serialVersionUID = 3L;

    private final String[] dimensionNames;

    private final boolean standardise;

    /**
     * Constructs a linear regression model trained via SGD.
     * @param name The model name.
     * @param dimensionNames The regression dimension names.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param parameters The model parameters.
     * @param standardise Is the model fitted on standardised regressors?
     */
    FMRegressionModel(String name, String[] dimensionNames, ModelProvenance provenance,
                      ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo,
                      FMParameters parameters, boolean standardise) {
        super(name, provenance, featureIDMap, outputIDInfo, parameters, false);
        this.dimensionNames = dimensionNames;
        this.standardise = standardise;
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        PredAndActive predTuple = predictSingle(example);
        double[] predictions = predTuple.prediction.toArray();
        if (standardise) {
            predictions = unstandardisePredictions(predictions);
        }
        return new Prediction<>(new Regressor(dimensionNames,predictions), predTuple.numActiveFeatures, example);
    }

    /**
     * Converts zero mean unit variance predictions into the true range.
     * @param predictions The predictions to convert.
     */
    private double[] unstandardisePredictions(double[] predictions) {
        ImmutableRegressionInfo info = (ImmutableRegressionInfo) outputIDInfo;
        for (int i = 0; i < predictions.length; i++) {
            double mean = info.getMean(i);
            double variance = info.getVariance(i);
            predictions[i] = (predictions[i] * variance) + mean;
        }
        return predictions;
    }

    @Override
    protected FMRegressionModel copy(String newName, ModelProvenance newProvenance) {
        return new FMRegressionModel(newName,Arrays.copyOf(dimensionNames,dimensionNames.length),newProvenance,featureIDMap,outputIDInfo,(FMParameters)modelParameters.copy(),standardise);
    }

    @Override
    protected String getDimensionName(int index) {
        return dimensionNames[index];
    }

    @Override
    public OnnxMl.ModelProto exportONNXModel(String domain, long modelVersion) {
        ONNXContext context = new ONNXContext();

        // Build graph
        OnnxMl.GraphProto graph = exportONNXGraph(context);

        return innerExportONNXModel(graph,domain,modelVersion);
    }

    @Override
    public OnnxMl.GraphProto exportONNXGraph(ONNXContext context) {
        OnnxMl.GraphProto.Builder graphBuilder = OnnxMl.GraphProto.newBuilder();
        graphBuilder.setName("FMMultiLabelModel");

        // Make inputs and outputs
        OnnxMl.TypeProto inputType = ONNXMathUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,featureIDMap.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto inputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(inputType).setName("input").build();
        graphBuilder.addInput(inputValueProto);
        String outputName = "output";
        OnnxMl.TypeProto outputType = ONNXMathUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,outputIDInfo.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto outputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(outputType).setName(outputName).build();
        graphBuilder.addOutput(outputValueProto);

        // Build the output neutral bits of the onnx graph
        String fmOutputName = generateONNXGraph(context, graphBuilder, inputValueProto.getName());

        if (standardise) {
            // standardise the FM output
            ImmutableRegressionInfo info = (ImmutableRegressionInfo) outputIDInfo;
            double[] means = new double[outputIDInfo.size()];
            double[] variances = new double[outputIDInfo.size()];
            for (int i = 0; i < means.length; i++) {
                means[i] = info.getMean(i);
                variances[i] = info.getVariance(i);
            }

            // Create mean and variance initializers
            OnnxMl.TensorProto outputMeanProto = ONNXMathUtils.arrayBuilder(context,context.generateUniqueName("y_mean"),means);
            graphBuilder.addInitializer(outputMeanProto);
            OnnxMl.TensorProto outputVarianceProto = ONNXMathUtils.arrayBuilder(context, context.generateUniqueName("y_var"),variances);
            graphBuilder.addInitializer(outputVarianceProto);

            // Add standardisation operations
            String varianceOutput = context.generateUniqueName("y_var_scale_output");
            OnnxMl.NodeProto varianceScale = ONNXOperators.MUL.build(context, new String[]{fmOutputName,outputVarianceProto.getName()}, varianceOutput);
            graphBuilder.addNode(varianceScale);
            OnnxMl.NodeProto meanScale = ONNXOperators.ADD.build(context, new String[]{varianceOutput,outputMeanProto.getName()}, outputName);
            graphBuilder.addNode(meanScale);
        } else {
            // Not standardised, so link up the FM output to the graph output
            OnnxMl.NodeProto output = ONNXOperators.IDENTITY.build(context, fmOutputName, outputName);
            graphBuilder.addNode(output);
        }

        return graphBuilder.build();
    }
}
