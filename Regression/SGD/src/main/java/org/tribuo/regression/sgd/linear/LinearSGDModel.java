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

package org.tribuo.regression.sgd.linear;

import ai.onnx.proto.OnnxMl;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXShape;
import org.tribuo.math.onnx.ONNXMathUtils;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;

import java.io.IOException;
import java.util.Arrays;

/**
 * The inference time version of a linear model trained using SGD.
 * The output dimensions are independent, unless they are tied together by the
 * optimiser.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public class LinearSGDModel extends AbstractLinearSGDModel<Regressor> implements ONNXExportable {
    private static final long serialVersionUID = 3L;

    private final String[] dimensionNames;

    // Unused as the weights now live in AbstractSGDModel
    // It remains for serialization compatibility with Tribuo 4.0
    @Deprecated
    private DenseMatrix weights = null;

    /**
     * Constructs a linear regression model trained via SGD.
     * @param name The model name.
     * @param dimensionNames The regression dimension names.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param parameters The model parameters (i.e., the weight matrix).
     */
    LinearSGDModel(String name, String[] dimensionNames, ModelProvenance provenance,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo,
                          LinearParameters parameters) {
        super(name, provenance, featureIDMap, outputIDInfo, parameters, false);
        this.dimensionNames = dimensionNames;
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        PredAndActive predTuple = predictSingle(example);
        return new Prediction<>(new Regressor(dimensionNames,predTuple.prediction.toArray()), predTuple.numActiveFeatures-1, example);
    }

    @Override
    protected LinearSGDModel copy(String newName, ModelProvenance newProvenance) {
        return new LinearSGDModel(newName,Arrays.copyOf(dimensionNames,dimensionNames.length),newProvenance,featureIDMap,outputIDInfo,(LinearParameters)modelParameters.copy());
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
        graphBuilder.setName("Regression-LinearSGDModel");

        // Make inputs and outputs
        OnnxMl.TypeProto inputType = ONNXMathUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,featureIDMap.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto inputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(inputType).setName("input").build();
        graphBuilder.addInput(inputValueProto);
        OnnxMl.TypeProto outputType = ONNXMathUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,outputIDInfo.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto outputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(outputType).setName("output").build();
        graphBuilder.addOutput(outputValueProto);

        // Add weights
        OnnxMl.TensorProto weightInitializerProto = weightBuilder(context);
        graphBuilder.addInitializer(weightInitializerProto);

        // Add biases
        OnnxMl.TensorProto biasInitializerProto = biasBuilder(context);
        graphBuilder.addInitializer(biasInitializerProto);

        // Make gemm
        String[] gemmInputs = new String[]{inputValueProto.getName(),weightInitializerProto.getName(),biasInitializerProto.getName()};
        OnnxMl.NodeProto gemm = ONNXOperators.GEMM.build(context,gemmInputs,"output");
        graphBuilder.addNode(gemm);

        return graphBuilder.build();
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        // Bounce old 4.0 style models into the new 4.1 style models
        if (weights != null && modelParameters == null) {
            modelParameters = new LinearParameters(weights);
            weights = null;
            addBias = true;
        }
    }
}
