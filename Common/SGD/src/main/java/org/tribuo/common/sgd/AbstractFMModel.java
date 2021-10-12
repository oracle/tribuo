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

package org.tribuo.common.sgd;

import ai.onnx.proto.OnnxMl;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Tribuo;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.provenance.ModelProvenance;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.PriorityQueue;

/**
 * A quadratic factorization machine model trained using SGD.
 * <p>
 * It's an {@link AbstractSGDTrainer} operating on {@link FMParameters}.
 * <p>
 * See:
 * <pre>
 * Rendle, S.
 * Factorization machines.
 * 2010 IEEE International Conference on Data Mining
 * </pre>
 */
public abstract class AbstractFMModel<T extends Output<T>> extends AbstractSGDModel<T> {
    private static final long serialVersionUID = 1L;

    /**
     * Constructs a factorization machine model trained via SGD.
     *
     * @param name                   The model name.
     * @param provenance             The model provenance.
     * @param featureIDMap           The feature domain.
     * @param outputIDInfo           The output domain.
     * @param parameters             The model parameters.
     * @param generatesProbabilities Does this model generate probabilities?
     */
    protected AbstractFMModel(String name, ModelProvenance provenance,
                              ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                              FMParameters parameters, boolean generatesProbabilities) {
        super(name, provenance, featureIDMap, outputIDInfo, parameters, generatesProbabilities, false);
    }

    /**
     * Gets the top {@code n} features for each output dimension.
     * <p>
     * Note that the feature rankings are based only off the linear portion of the
     * factorization machine.
     * @param n The number of features to return. If this value is less than 0,
     * all features are returned for each class.
     * @return A map from string outputs to an ordered list of pairs of
     * feature names and weights associated with that feature in the factorization machine.
     */
    // TODO investigate using the factorized representation magnitude as an additional feature weight.
    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        DenseVector biases = (DenseVector) modelParameters.get()[0];
        DenseMatrix baseWeights = (DenseMatrix) modelParameters.get()[1];
        int maxFeatures = n < 0 ? featureIDMap.size() + 1 : n;

        Comparator<Pair<String, Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));

        //
        // Use a priority queue to find the top N features.
        int numClasses = baseWeights.getDimension1Size();
        int numFeatures = baseWeights.getDimension2Size();
        Map<String, List<Pair<String, Double>>> map = new HashMap<>();
        for (int i = 0; i < numClasses; i++) {
            PriorityQueue<Pair<String, Double>> q = new PriorityQueue<>(maxFeatures, comparator);

            for (int j = 0; j < numFeatures; j++) {
                Pair<String, Double> curr = new Pair<>(featureIDMap.get(j).getName(), baseWeights.get(i, j));

                if (q.size() < maxFeatures) {
                    q.offer(curr);
                } else if (comparator.compare(curr, q.peek()) > 0) {
                    q.poll();
                    q.offer(curr);
                }
            }
            Pair<String, Double> curr = new Pair<>(BIAS_FEATURE, biases.get(i));

            if (q.size() < maxFeatures) {
                q.offer(curr);
            } else if (comparator.compare(curr, q.peek()) > 0) {
                q.poll();
                q.offer(curr);
            }
            List<Pair<String, Double>> b = new ArrayList<>();
            while (q.size() > 0) {
                b.add(q.poll());
            }

            Collections.reverse(b);
            map.put(getDimensionName(i), b);
        }
        return map;
    }

    /**
     * Returns a copy of the linear weights.
     *
     * @return The linear weights.
     */
    public DenseMatrix getLinearWeightsCopy() {
        return ((DenseMatrix) modelParameters.get()[1]).copy();
    }

    /**
     * Returns a copy of the output dimension biases.
     *
     * @return The biases.
     */
    public DenseVector getBiasesCopy() {
        return ((DenseVector) modelParameters.get()[0]).copy();
    }

    /**
     * Returns a copy of the factors.
     * There is one factor matrix per output dimension.
     * The first factor matrix dimension is the factor dimension,
     * the second is the number of features.
     *
     * @return The factors.
     */
    public Tensor[] getFactorsCopy() {
        Tensor[] params = modelParameters.get();
        Tensor[] paramCopy = new Tensor[params.length - 2];
        for (int i = 0; i < paramCopy.length; i++) {
            paramCopy[i] = params[i + 2].copy();
        }
        return paramCopy;
    }

    /**
     * Factorization machines don't provide excuses, use an explainer.
     *
     * @param example The input example.
     * @return Optional.empty.
     */
    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        return Optional.empty();
    }

    /**
     * Gets the name of the indexed output dimension.
     *
     * @param index The output dimension index.
     * @return The name of the requested output dimension.
     */
    protected abstract String getDimensionName(int index);

    /**
     * Builds the ModelProto according to the standards for this model.
     *
     * @param graph        The model graph.
     * @param domain       The model domain string.
     * @param modelVersion The model version number.
     * @return The ModelProto.
     */
    protected OnnxMl.ModelProto innerExportONNXModel(OnnxMl.GraphProto graph, String domain, long modelVersion) {
        // Build model
        OnnxMl.ModelProto.Builder builder = OnnxMl.ModelProto.newBuilder();
        builder.setGraph(graph);
        builder.setDomain(domain);
        builder.setProducerName("Tribuo");
        builder.setProducerVersion(Tribuo.VERSION);
        builder.setModelVersion(modelVersion);
        builder.setDocString(toString());
        builder.addOpsetImport(ONNXOperators.getOpsetProto());
        builder.setIrVersion(6);
        return builder.build();
    }

    /**
     * Builds a TensorProto containing the supplied DenseMatrix.
     *
     * @param context   The ONNX context for naming.
     * @param name      The name for this tensor proto.
     * @param matrix    The matrix to store.
     * @param transpose Should the matrix be transposed into the tensor?
     * @return The matrix TensorProto.
     */
    protected static OnnxMl.TensorProto matrixBuilder(ONNXContext context, String name, DenseMatrix matrix, boolean transpose) {
        OnnxMl.TensorProto.Builder matrixBuilder = OnnxMl.TensorProto.newBuilder();
        matrixBuilder.setName(context.generateUniqueName(name));
        int dim1, dim2;
        if (transpose) {
            dim1 = matrix.getDimension2Size();
            dim2 = matrix.getDimension1Size();
        } else {
            dim1 = matrix.getDimension1Size();
            dim2 = matrix.getDimension2Size();
        }
        matrixBuilder.addDims(dim1);
        matrixBuilder.addDims(dim2);
        matrixBuilder.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
        ByteBuffer buffer = ByteBuffer.allocate(dim1 * dim2 * 4).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                if (transpose) {
                    floatBuffer.put((float) matrix.get(j, i));
                } else {
                    floatBuffer.put((float) matrix.get(i, j));
                }
            }
        }
        floatBuffer.rewind();
        matrixBuilder.setRawData(ByteString.copyFrom(buffer));
        return matrixBuilder.build();
    }

    /**
     * Builds a TensorProto containing the supplied dense vector.
     *
     * @param context The ONNX context for naming.
     * @param name    The name for this tensor proto.
     * @param vector  The vector to store.
     * @return The vector TensorProto.
     */
    protected static OnnxMl.TensorProto vectorBuilder(ONNXContext context, String name, DenseVector vector) {
        OnnxMl.TensorProto.Builder vectorBuilder = OnnxMl.TensorProto.newBuilder();
        vectorBuilder.setName(context.generateUniqueName(name));
        vectorBuilder.addDims(vector.size());
        vectorBuilder.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
        ByteBuffer buffer = ByteBuffer.allocate(vector.size() * 4).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        for (int i = 0; i < vector.size(); i++) {
            floatBuffer.put((float) vector.get(i));
        }
        floatBuffer.rewind();
        vectorBuilder.setRawData(ByteString.copyFrom(buffer));
        return vectorBuilder.build();
    }

    /**
     * Constructs the shared stem of the Factorization Machine, used by all output types.
     * <p>
     * Writes into the supplied graph builder.
     *
     * @param context      The ONNX context.
     * @param graphBuilder The graph builder.
     * @return The name of the output.
     */
    protected String generateONNXGraph(ONNXContext context, OnnxMl.GraphProto.Builder graphBuilder, String inputName) {
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
        OnnxMl.TensorProto biasInitializerProto = vectorBuilder(context, "fm_biases", (DenseVector) modelParams[0]);
        graphBuilder.addInitializer(biasInitializerProto);

        // Add embedding vectors
        OnnxMl.TensorProto[] embeddingProtos = new OnnxMl.TensorProto[outputIDInfo.size()];
        for (int i = 0; i < outputIDInfo.size(); i++) {
            embeddingProtos[i] = matrixBuilder(context, "fm_embedding_" + i, (DenseMatrix) modelParams[i + 2], true);
            graphBuilder.addInitializer(embeddingProtos[i]);
        }

        // Make gemm
        String[] gemmInputs = new String[]{inputName, weightInitializerProto.getName(), biasInitializerProto.getName()};
        OnnxMl.NodeProto gemm = ONNXOperators.GEMM.build(context, gemmInputs, context.generateUniqueName("gemm_output"));
        graphBuilder.addNode(gemm);

        // Make feature pow
        OnnxMl.NodeProto featurePow = ONNXOperators.POW.build(context, new String[]{inputName, twoConst.getName()},
                context.generateUniqueName("feature_pow"));
        graphBuilder.addNode(featurePow);

        // Make interaction terms
        String[] embeddingOutputs = new String[outputIDInfo.size()];
        for (int i = 0; i < outputIDInfo.size(); i++) {
            // Feature matrix * embedding matrix = batch_size, embedding dim
            OnnxMl.NodeProto gemmFeatureEmb = ONNXOperators.GEMM.build(context,
                    new String[]{inputName, embeddingProtos[i].getName()},
                    context.generateUniqueName("gemm_input_emb"));
            graphBuilder.addNode(gemmFeatureEmb);
            // Square the output
            OnnxMl.NodeProto powFeatureEmb = ONNXOperators.POW.build(context,
                    new String[]{gemmFeatureEmb.getOutput(0), twoConst.getName()},
                    context.generateUniqueName("pow_input_emb"));
            graphBuilder.addNode(powFeatureEmb);
            // Square the embeddings
            OnnxMl.NodeProto powEmb = ONNXOperators.POW.build(context,
                    new String[]{embeddingProtos[i].getName(), twoConst.getName()},
                    context.generateUniqueName("pow_emb"));
            graphBuilder.addNode(powEmb);
            // squared features * squared embeddings
            OnnxMl.NodeProto gemmSquaredFeatureSquaredEmb = ONNXOperators.GEMM.build(context,
                    new String[]{featurePow.getOutput(0), powEmb.getOutput(0)},
                    context.generateUniqueName("gemm_squared_input_squared_emb"));
            graphBuilder.addNode(gemmSquaredFeatureSquaredEmb);
            // squared product subtract product of squares
            OnnxMl.NodeProto subtract = ONNXOperators.SUB.build(context,
                    new String[]{powFeatureEmb.getOutput(0), gemmSquaredFeatureSquaredEmb.getOutput(0)},
                    context.generateUniqueName("squared_prod_subtract_prod_of_squares"));
            graphBuilder.addNode(subtract);
            // sum over embedding dimensions
            OnnxMl.NodeProto sumOverEmbeddings = ONNXOperators.REDUCE_SUM.build(context,
                    subtract.getOutput(0),
                    context.generateUniqueName("sum_over_embeddings"),
                    Collections.singletonMap("axes", new int[]{1}));
            graphBuilder.addNode(sumOverEmbeddings);
            // Divide by 2
            OnnxMl.NodeProto scaledInteraction = ONNXOperators.DIV.build(context,
                    new String[]{sumOverEmbeddings.getOutput(0), twoConst.getName()},
                    context.generateUniqueName("scaled_interaction"));
            graphBuilder.addNode(scaledInteraction);
            // Store the output name
            embeddingOutputs[i] = scaledInteraction.getOutput(0);
        }

        // Make concat
        OnnxMl.NodeProto concat = ONNXOperators.CONCAT.build(context, embeddingOutputs, context.generateUniqueName("fm_concat"),
                Collections.singletonMap("axis", 1)
        );
        graphBuilder.addNode(concat);

        // Add to gemm
        OnnxMl.NodeProto addGemmConcat = ONNXOperators.ADD.build(context, new String[]{gemm.getOutput(0), concat.getOutput(0)}, context.generateUniqueName("fm_output"));
        graphBuilder.addNode(addGemmConcat);

        return addGemmConcat.getOutput(0);
    }

}
