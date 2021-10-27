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

package org.tribuo.classification.sgd.linear;

import ai.onnx.proto.OnnxMl;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXShape;
import org.tribuo.math.onnx.ONNXMathUtils;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The inference time version of a linear model trained using SGD.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public class LinearSGDModel extends AbstractLinearSGDModel<Label> implements ONNXExportable {
    private static final long serialVersionUID = 2L;

    private final VectorNormalizer normalizer;

    // Unused as the weights now live in AbstractSGDModel
    // It remains for serialization compatibility with Tribuo 4.0
    @Deprecated
    private DenseMatrix weights = null;

    /**
     * Constructs a linear classification model trained via SGD.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param parameters The model parameters (i.e., the weight matrix).
     * @param normalizer The normalization function.
     * @param generatesProbabilities Does this model generate probabilities?
     */
    LinearSGDModel(String name, ModelProvenance provenance,
                   ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo,
                   LinearParameters parameters, VectorNormalizer normalizer, boolean generatesProbabilities) {
        super(name, provenance, featureIDMap, outputIDInfo, parameters, generatesProbabilities);
        this.normalizer = normalizer;
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        PredAndActive predTuple = predictSingle(example);
        DenseVector prediction = predTuple.prediction;
        prediction.normalize(normalizer);

        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String,Label> predMap = new LinkedHashMap<>();
        for (int i = 0; i < prediction.size(); i++) {
            String labelName = outputIDInfo.getOutput(i).getLabel();
            double score = prediction.get(i);
            Label label = new Label(labelName, score);
            predMap.put(labelName,label);
            if (score > maxScore) {
                maxScore = score;
                maxLabel = label;
            }
        }
        return new Prediction<>(maxLabel, predMap, predTuple.numActiveFeatures-1, example, generatesProbabilities);
    }

    @Override
    protected LinearSGDModel copy(String newName, ModelProvenance newProvenance) {
        return new LinearSGDModel(newName,newProvenance,featureIDMap,outputIDInfo,(LinearParameters)modelParameters.copy(),normalizer,generatesProbabilities);
    }

    @Override
    protected String getDimensionName(int index) {
        return outputIDInfo.getOutput(index).getLabel();
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
        graphBuilder.setName("Classification-LinearSGDModel");

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
        OnnxMl.NodeProto gemm = ONNXOperators.GEMM.build(context,gemmInputs,context.generateUniqueName("gemm_output"));
        graphBuilder.addNode(gemm);

        // Make output normalizer
        List<OnnxMl.NodeProto> normalizerProtos = normalizer.exportNormalizer(context,gemm.getOutput(0),"output");
        if (normalizerProtos.isEmpty()) {
            throw new IllegalArgumentException("Normalizer " + normalizer.getClass() + " cannot be exported in ONNX models.");
        } else {
            graphBuilder.addAllNode(normalizerProtos);
        }

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
