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
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Tribuo;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.onnx.ONNXMathUtils;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXUtils;
import org.tribuo.provenance.ModelProvenance;

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

        // Extract provenance and store in metadata
        OnnxMl.StringStringEntryProto.Builder metaBuilder = OnnxMl.StringStringEntryProto.newBuilder();
        metaBuilder.setKey(ONNXExportable.PROVENANCE_METADATA_FIELD);
        String serializedProvenance = ONNXExportable.SERIALIZER.marshalAndSerialize(getProvenance());
        metaBuilder.setValue(serializedProvenance);
        builder.addMetadataProps(metaBuilder.build());

        return builder.build();
    }

    /**
     * Constructs the shared stem of the Factorization Machine, used by all output types.
     * <p>
     * Writes into the supplied context.
     *
     * @param context      The ONNX context.
     * @return The name of the output.
     */
    protected String generateONNXGraph(ONNXContext context, String inputName) {
        Tensor[] modelParams = modelParameters.get();

        // Add constants
        OnnxMl.TensorProto twoConst = OnnxMl.TensorProto.newBuilder()
                .setName(context.generateUniqueName("two_const"))
                .setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber())
                .addFloatData(2.0f)
                .build();
        context.addInitializer(twoConst);

        // Add weights
        OnnxMl.TensorProto weightInitializerProto = ONNXMathUtils.floatMatrixBuilder(context, "fm_linear_weights", (Matrix) modelParams[1], true);
        context.addInitializer(weightInitializerProto);

        // Add biases
        OnnxMl.TensorProto biasInitializerProto = ONNXMathUtils.floatVectorBuilder(context, "fm_biases", (SGDVector) modelParams[0]);
        context.addInitializer(biasInitializerProto);

        // Add embedding vectors
        OnnxMl.TensorProto[] embeddingProtos = new OnnxMl.TensorProto[outputIDInfo.size()];
        for (int i = 0; i < outputIDInfo.size(); i++) {
            embeddingProtos[i] = ONNXMathUtils.floatMatrixBuilder(context, "fm_embedding_" + i, (Matrix) modelParams[i + 2], true);
            context.addInitializer(embeddingProtos[i]);
        }

        // Make gemm
        String[] gemmInputs = new String[]{inputName, weightInitializerProto.getName(), biasInitializerProto.getName()};
        OnnxMl.NodeProto gemm = ONNXOperators.GEMM.build(context, gemmInputs, context.generateUniqueName("gemm_output"));
        context.addNode(gemm);

        // Make feature pow
        OnnxMl.NodeProto featurePow = ONNXOperators.POW.build(context, new String[]{inputName, twoConst.getName()},
                context.generateUniqueName("feature_pow"));
        context.addNode(featurePow);

        // Make interaction terms
        String[] embeddingOutputs = new String[outputIDInfo.size()];
        for (int i = 0; i < outputIDInfo.size(); i++) {
            // Feature matrix * embedding matrix = batch_size, embedding dim
            OnnxMl.NodeProto gemmFeatureEmb = ONNXOperators.GEMM.build(context,
                    new String[]{inputName, embeddingProtos[i].getName()},
                    context.generateUniqueName("gemm_input_emb"));
            context.addNode(gemmFeatureEmb);
            // Square the output
            OnnxMl.NodeProto powFeatureEmb = ONNXOperators.POW.build(context,
                    new String[]{gemmFeatureEmb.getOutput(0), twoConst.getName()},
                    context.generateUniqueName("pow_input_emb"));
            context.addNode(powFeatureEmb);
            // Square the embeddings
            OnnxMl.NodeProto powEmb = ONNXOperators.POW.build(context,
                    new String[]{embeddingProtos[i].getName(), twoConst.getName()},
                    context.generateUniqueName("pow_emb"));
            context.addNode(powEmb);
            // squared features * squared embeddings
            OnnxMl.NodeProto gemmSquaredFeatureSquaredEmb = ONNXOperators.GEMM.build(context,
                    new String[]{featurePow.getOutput(0), powEmb.getOutput(0)},
                    context.generateUniqueName("gemm_squared_input_squared_emb"));
            context.addNode(gemmSquaredFeatureSquaredEmb);
            // squared product subtract product of squares
            OnnxMl.NodeProto subtract = ONNXOperators.SUB.build(context,
                    new String[]{powFeatureEmb.getOutput(0), gemmSquaredFeatureSquaredEmb.getOutput(0)},
                    context.generateUniqueName("squared_prod_subtract_prod_of_squares"));
            context.addNode(subtract);
            // sum over embedding dimensions
            OnnxMl.TensorProto sumAxes = ONNXUtils.arrayBuilder(context,"sum_over_embeddings_axes",new long[]{1});
            context.addInitializer(sumAxes);
            OnnxMl.NodeProto sumOverEmbeddings = ONNXOperators.REDUCE_SUM.build(context,
                    new String[]{subtract.getOutput(0),sumAxes.getName()},
                    context.generateUniqueName("sum_over_embeddings"));
            context.addNode(sumOverEmbeddings);
            // Divide by 2
            OnnxMl.NodeProto scaledInteraction = ONNXOperators.DIV.build(context,
                    new String[]{sumOverEmbeddings.getOutput(0), twoConst.getName()},
                    context.generateUniqueName("scaled_interaction"));
            context.addNode(scaledInteraction);
            // Store the output name
            embeddingOutputs[i] = scaledInteraction.getOutput(0);
        }

        // Make concat
        OnnxMl.NodeProto concat = ONNXOperators.CONCAT.build(context, embeddingOutputs, context.generateUniqueName("fm_concat"),
                Collections.singletonMap("axis", 1)
        );
        context.addNode(concat);

        // Add to gemm
        OnnxMl.NodeProto addGemmConcat = ONNXOperators.ADD.build(context, new String[]{gemm.getOutput(0), concat.getOutput(0)}, context.generateUniqueName("fm_output"));
        context.addNode(addGemmConcat);

        return addGemmConcat.getOutput(0);
    }

}
