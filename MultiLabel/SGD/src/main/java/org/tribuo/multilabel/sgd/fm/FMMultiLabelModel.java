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

package org.tribuo.multilabel.sgd.fm;

import ai.onnx.proto.OnnxMl;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.common.sgd.AbstractFMModel;
import org.tribuo.common.sgd.FMParameters;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.onnx.ONNXShape;
import org.tribuo.math.onnx.ONNXMathUtils;
import org.tribuo.provenance.ModelProvenance;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The inference time version of a multi-label factorization machine trained using SGD.
 * <p>
 * See:
 * <pre>
 * Rendle, S.
 * Factorization machines.
 * 2010 IEEE International Conference on Data Mining
 * </pre>
 */
public class FMMultiLabelModel extends AbstractFMModel<MultiLabel> implements ONNXExportable {
    private static final long serialVersionUID = 2L;

    private final VectorNormalizer normalizer;
    private final double threshold;

    /**
     * Constructs a linear regression model trained via SGD.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param parameters The model parameters (i.e., the weight matrix).
     * @param normalizer The output normalizer (usually sigmoid or no-op).
     * @param generatesProbabilities Does this model produce probabilistic outputs.
     * @param threshold The threshold for emitting a label.
     */
    FMMultiLabelModel(String name, ModelProvenance provenance,
                      ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<MultiLabel> outputIDInfo,
                      FMParameters parameters, VectorNormalizer normalizer, boolean generatesProbabilities, double threshold) {
        super(name, provenance, featureIDMap, outputIDInfo, parameters, generatesProbabilities);
        this.normalizer = normalizer;
        this.threshold = threshold;
    }

    @Override
    public Prediction<MultiLabel> predict(Example<MultiLabel> example) {
        PredAndActive predTuple = predictSingle(example);
        DenseVector outputs = predTuple.prediction;
        outputs.normalize(normalizer);
        Map<String,MultiLabel> fullLabels = new HashMap<>();
        Set<Label> predictedLabels = new HashSet<>();
        for (int i = 0; i < outputs.size(); i++) {
            String labelName = outputIDInfo.getOutput(i).getLabelString();
            double labelScore = outputs.get(i);
            Label score = new Label(outputIDInfo.getOutput(i).getLabelString(),labelScore);
            if (labelScore > threshold) {
                predictedLabels.add(score);
            }
            fullLabels.put(labelName,new MultiLabel(score));
        }
        return new Prediction<>(new MultiLabel(predictedLabels), fullLabels, predTuple.numActiveFeatures - 1, example, generatesProbabilities);
    }

    @Override
    protected String getDimensionName(int index) {
        return outputIDInfo.getOutput(index).getLabelString();
    }

    @Override
    protected FMMultiLabelModel copy(String newName, ModelProvenance newProvenance) {
        return new FMMultiLabelModel(newName,newProvenance,featureIDMap,outputIDInfo,(FMParameters)modelParameters.copy(),normalizer,generatesProbabilities,threshold);
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
        OnnxMl.TypeProto outputType = ONNXMathUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,outputIDInfo.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto outputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(outputType).setName("output").build();
        graphBuilder.addOutput(outputValueProto);

        // Build the output neutral bits of the onnx graph
        String outputName = generateONNXGraph(context, graphBuilder, inputValueProto.getName());

        // Make output normalizer
        List<OnnxMl.NodeProto> normalizerProtos = normalizer.exportNormalizer(context,outputName,"output");
        if (normalizerProtos.isEmpty()) {
            throw new IllegalArgumentException("Normalizer " + normalizer.getClass() + " cannot be exported in ONNX models.");
        } else {
            graphBuilder.addAllNode(normalizerProtos);
        }

        return graphBuilder.build();
    }
}
