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

package org.tribuo.classification.sgd.fm;

import ai.onnx.proto.OnnxMl;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.common.sgd.AbstractFMModel;
import org.tribuo.common.sgd.FMParameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXShape;
import org.tribuo.onnx.ONNXUtils;
import org.tribuo.provenance.ModelProvenance;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The inference time version of a factorization machine trained using SGD.
 * <p>
 * See:
 * <pre>
 * Rendle, S.
 * Factorization machines.
 * 2010 IEEE International Conference on Data Mining
 * </pre>
 */
public class FMClassificationModel extends AbstractFMModel<Label> implements ONNXExportable {
    private static final long serialVersionUID = 1L;

    private final VectorNormalizer normalizer;

    /**
     * Constructs a classification factorization machine trained via SGD.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param parameters The model parameters.
     * @param normalizer The normalization function.
     * @param generatesProbabilities Does this model generate probabilities?
     */
    FMClassificationModel(String name, ModelProvenance provenance,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo,
                          FMParameters parameters, VectorNormalizer normalizer, boolean generatesProbabilities) {
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
        return new Prediction<>(maxLabel, predMap, predTuple.numActiveFeatures, example, generatesProbabilities);
    }

    @Override
    protected FMClassificationModel copy(String newName, ModelProvenance newProvenance) {
        return new FMClassificationModel(newName,newProvenance,featureIDMap,outputIDInfo,(FMParameters)modelParameters.copy(),normalizer,generatesProbabilities);
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
        graphBuilder.setName("FMClassificationModel");

        // Make inputs and outputs
        OnnxMl.TypeProto inputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,featureIDMap.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto inputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(inputType).setName("input").build();
        graphBuilder.addInput(inputValueProto);
        OnnxMl.TypeProto outputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,outputIDInfo.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto outputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(outputType).setName("output").build();
        graphBuilder.addOutput(outputValueProto);

        Tensor[] modelParams = modelParameters.get();

        // Add constants
        OnnxMl.TensorProto twoConst = OnnxMl.TensorProto.newBuilder()
                .setName(context.generateUniqueName("two_const"))
                .setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber())
                .addFloatData(2.0f)
                .build();
        graphBuilder.addInitializer(twoConst);

        // Add weights
        OnnxMl.TensorProto weightInitializerProto = matrixBuilder(context, "fm_linear_weights", (DenseMatrix) modelParams[1], true);
        graphBuilder.addInitializer(weightInitializerProto);

        // Add biases
        OnnxMl.TensorProto biasInitializerProto = biasBuilder(context);
        graphBuilder.addInitializer(biasInitializerProto);

        // Add embedding vectors
        OnnxMl.TensorProto[] embeddingProtos = new OnnxMl.TensorProto[outputIDInfo.size()];
        for (int i = 0; i < outputIDInfo.size(); i++) {
            embeddingProtos[i] = matrixBuilder(context, "fm_embedding_"+i, (DenseMatrix) modelParams[i+2], true);
            graphBuilder.addInitializer(embeddingProtos[i]);
        }

        // Make gemm
        String[] gemmInputs = new String[]{inputValueProto.getName(),weightInitializerProto.getName(),biasInitializerProto.getName()};
        OnnxMl.NodeProto gemm = ONNXOperators.GEMM.build(context,gemmInputs,context.generateUniqueName("gemm_output"));
        graphBuilder.addNode(gemm);

        // Make feature pow
        OnnxMl.NodeProto featurePow = ONNXOperators.POW.build(context,new String[]{inputValueProto.getName(),twoConst.getName()},
                context.generateUniqueName("feature_pow"));
        graphBuilder.addNode(featurePow);

        // Make interaction terms
        String[] embeddingOutputs = new String[outputIDInfo.size()];
        for (int i = 0; i < outputIDInfo.size(); i++) {
            // Feature matrix * embedding matrix = batch_size, embedding dim
            OnnxMl.NodeProto gemmFeatureEmb = ONNXOperators.GEMM.build(context,
                    new String[]{inputValueProto.getName(),embeddingProtos[i].getName()},
                    context.generateUniqueName("gemm_input_emb"),
                    Collections.singletonMap("transB",1));
            graphBuilder.addNode(gemmFeatureEmb);
            // Square the output
            OnnxMl.NodeProto powFeatureEmb = ONNXOperators.POW.build(context,
                    new String[]{gemmFeatureEmb.getOutput(0),twoConst.getName()},
                    context.generateUniqueName("pow_input_emb"));
            graphBuilder.addNode(powFeatureEmb);
            // Square the embeddings
            OnnxMl.NodeProto powEmb = ONNXOperators.POW.build(context,
                    new String[]{embeddingProtos[i].getName(),twoConst.getName()},
                    context.generateUniqueName("pow_emb"));
            graphBuilder.addNode(powEmb);
            // squared features * squared embeddings
            OnnxMl.NodeProto gemmSquaredFeatureSquaredEmb = ONNXOperators.GEMM.build(context,
                    new String[]{featurePow.getOutput(0),powEmb.getOutput(0)},
                    context.generateUniqueName("gemm_squared_input_squared_emb"),
                    Collections.singletonMap("transB",1));
            graphBuilder.addNode(gemmSquaredFeatureSquaredEmb);
            // squared product subtract product of squares
            OnnxMl.NodeProto subtract = ONNXOperators.SUB.build(context,
                    new String[]{powFeatureEmb.getOutput(0),gemmSquaredFeatureSquaredEmb.getOutput(0)},
                    context.generateUniqueName("squared_prod_subtract_prod_of_squares"));
            graphBuilder.addNode(subtract);
            // sum over embedding dimensions
            OnnxMl.NodeProto sumOverEmbeddings = ONNXOperators.REDUCE_SUM.build(context,
                    subtract.getOutput(0),
                    context.generateUniqueName("sum_over_embeddings"),
                    Collections.singletonMap("axes",new int[]{1}));
            graphBuilder.addNode(sumOverEmbeddings);
            // Divide by 2
            OnnxMl.NodeProto scaledInteraction = ONNXOperators.DIV.build(context,
                    new String[]{sumOverEmbeddings.getOutput(0),twoConst.getName()},
                    context.generateUniqueName("scaled_interaction"));
            graphBuilder.addNode(scaledInteraction);
            // Store the output name
            embeddingOutputs[i] = scaledInteraction.getOutput(0);
        }

        // Make concat
        OnnxMl.NodeProto concat = ONNXOperators.CONCAT.build(context, embeddingOutputs, context.generateUniqueName("fm_concat"),
                    Collections.singletonMap("axis",1)
                );
        graphBuilder.addNode(concat);

        // Add to gemm
        OnnxMl.NodeProto addGemmConcat = ONNXOperators.ADD.build(context, new String[]{gemm.getOutput(0),concat.getOutput(0)}, context.generateUniqueName("fm_output"));
        graphBuilder.addNode(addGemmConcat);

        // Make output normalizer
        List<OnnxMl.NodeProto> normalizerProtos = normalizer.exportNormalizer(context,addGemmConcat.getOutput(0),"output");
        if (normalizerProtos.isEmpty()) {
            throw new IllegalArgumentException("Normalizer " + normalizer.getClass() + " cannot be exported in ONNX models.");
        } else {
            graphBuilder.addAllNode(normalizerProtos);
        }

        return graphBuilder.build();
    }

}
