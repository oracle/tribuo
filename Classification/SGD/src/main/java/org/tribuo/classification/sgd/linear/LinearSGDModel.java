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

import com.google.protobuf.ByteString;
import ai.onnx.proto.OnnxMl;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.Tribuo;
import org.tribuo.classification.Label;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.util.ExpNormalizer;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXExport;
import org.tribuo.onnx.ONNXShape;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.LinkedHashMap;
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
public class LinearSGDModel extends AbstractLinearSGDModel<Label> implements ONNXExport {
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

        // Build model
        OnnxMl.ModelProto.Builder builder = OnnxMl.ModelProto.newBuilder();
        builder.setGraph(graph);
        builder.setDomain(domain);
        builder.setProducerName("Tribuo");
        builder.setProducerVersion(Tribuo.VERSION);
        builder.setModelVersion(modelVersion);
        builder.setDocString(toString());
        builder.addOpsetImport(OnnxMl.OperatorSetIdProto.newBuilder().setVersion(9).build());
        builder.setIrVersion(6);
        return builder.build();
    }

    @Override
    public OnnxMl.GraphProto exportONNXGraph(ONNXContext context) {
        OnnxMl.GraphProto.Builder graphBuilder = OnnxMl.GraphProto.newBuilder();

        // Make inputs and outputs
        OnnxMl.TypeProto inputType = buildTensorTypeNode(new ONNXShape(new long[]{-1,featureIDMap.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto inputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(inputType).setName("input").build();
        graphBuilder.addInput(inputValueProto);
        OnnxMl.TypeProto outputType = buildTensorTypeNode(new ONNXShape(new long[]{-1,outputIDInfo.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto outputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(outputType).setName("output").build();
        graphBuilder.addOutput(outputValueProto);

        // Make weights
        DenseMatrix weightMatrix = (DenseMatrix)modelParameters.get()[0];
        OnnxMl.TensorProto.Builder weightBuilder = OnnxMl.TensorProto.newBuilder();
        weightBuilder.setName(context.generateUniqueName("linear_sgd_weights"));
        weightBuilder.addDims(featureIDMap.size());
        weightBuilder.addDims(outputIDInfo.size());
        weightBuilder.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
        ByteBuffer buffer = ByteBuffer.allocate(featureIDMap.size()*outputIDInfo.size()*4).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        for (int j = 0; j < weightMatrix.getDimension2Size()-1; j++) {
            for (int i = 0; i < weightMatrix.getDimension1Size(); i++) {
                floatBuffer.put((float)weightMatrix.get(i,j));
            }
        }
        floatBuffer.rewind();
        weightBuilder.setRawData(ByteString.copyFrom(buffer));
        OnnxMl.TensorProto weightInitializerProto = weightBuilder.build();
        graphBuilder.addInitializer(weightInitializerProto);

        // Make biases
        OnnxMl.TensorProto.Builder biasBuilder = OnnxMl.TensorProto.newBuilder();
        biasBuilder.setName(context.generateUniqueName("linear_sgd_biases"));
        biasBuilder.addDims(outputIDInfo.size());
        biasBuilder.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
        ByteBuffer biasBuffer = ByteBuffer.allocate(outputIDInfo.size()*4).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer floatBiasBuffer = biasBuffer.asFloatBuffer();
        for (int i = 0; i < weightMatrix.getDimension1Size(); i++) {
            floatBiasBuffer.put((float)weightMatrix.get(i,weightMatrix.getDimension2Size()-1));
        }
        floatBiasBuffer.rewind();
        biasBuilder.setRawData(ByteString.copyFrom(biasBuffer));
        OnnxMl.TensorProto biasInitializerProto = biasBuilder.build();
        graphBuilder.addInitializer(biasInitializerProto);

        // Make gemm
        String[] gemmInputs = new String[]{inputValueProto.getName(),weightInitializerProto.getName(),biasInitializerProto.getName()};
        OnnxMl.NodeProto gemm = ONNXOperators.GEMM.build(context,gemmInputs,new String[]{context.generateUniqueName("gemm_output")},Collections.emptyMap());
        graphBuilder.addNode(gemm);

        // Make output normalizer
        if (!(normalizer instanceof ExpNormalizer)) {
            throw new IllegalStateException("Only works on softmax");
        }
        OnnxMl.NodeProto softmax = ONNXOperators.SOFTMAX.build(context,new String[]{gemm.getOutput(0)},new String[]{"output"}, Collections.singletonMap("axis",1));
        graphBuilder.addNode(softmax);

        return graphBuilder.build();
    }

    private static OnnxMl.TypeProto buildTensorTypeNode(ONNXShape shape, OnnxMl.TensorProto.DataType type) {
        OnnxMl.TypeProto.Builder builder = OnnxMl.TypeProto.newBuilder();

        OnnxMl.TypeProto.Tensor.Builder tensorBuilder = OnnxMl.TypeProto.Tensor.newBuilder();
        tensorBuilder.setElemType(type.getNumber());
        tensorBuilder.setShape(shape.getProto());
        builder.setTensorType(tensorBuilder.build());

        return builder.build();
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
