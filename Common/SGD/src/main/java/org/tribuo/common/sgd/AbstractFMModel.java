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
import org.tribuo.ONNXExportable;
import org.tribuo.Output;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.onnx.ONNXMathUtils;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.util.onnx.ONNXContext;
import org.tribuo.util.onnx.ONNXInitializer;
import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXOperators;
import org.tribuo.util.onnx.ONNXPlaceholder;
import org.tribuo.util.onnx.ONNXRef;

import java.util.ArrayList;
import java.util.Arrays;
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
     * Takes the unnormalized ONNX output of this model and applies an appropriate normalizer from the concrete class.
     * @param input Unnormalized ONNX leaf node.
     * @return Normalized ONNX leaf node.
     */
    protected abstract ONNXNode onnxOutput(ONNXNode input);

    /**
     * @return Name to write into the ONNX Model.
     */
    protected abstract String onnxModelName();

    /**
     * Writes this {@link org.tribuo.Model} into {@link OnnxMl.GraphProto.Builder} inside the input's
     * {@link ONNXContext}.
     * @param input The input to the model graph.
     * @return the output node of the model graph.
     */
    public ONNXNode writeONNXGraph(ONNXRef<?> input) {
        ONNXContext onnx = input.onnxContext();
        Tensor[] modelParams = modelParameters.get();

        ONNXInitializer twoConst = onnx.constant("two_const", 2.0f);
        ONNXInitializer sumAxes = onnx.array("sum_over_embedding_axes", new long[]{1});

        ONNXInitializer weights = ONNXMathUtils.floatMatrix(onnx, "fm_linear_weights", (Matrix) modelParams[1], true);
        ONNXInitializer bias = ONNXMathUtils.floatVector(onnx, "fm_biases", (SGDVector) modelParams[0]);

        // Make gemm
        ONNXNode gemm = input.apply(ONNXOperators.GEMM, Arrays.asList(weights, bias));

        // Make feature pow
        ONNXNode inputSquared = input.apply(ONNXOperators.POW, twoConst);

        List<ONNXNode> embeddingOutputs = new ArrayList<>();
        for(int i = 0; i < outputIDInfo.size(); i++) {

            // Embedding Weights
            ONNXInitializer embWeight = ONNXMathUtils.floatMatrix(onnx, "fm_embedding_" + i, (Matrix) modelParams[i + 2], true);

            // Feature matrix * embedding matrix = batch_size, embedding dim
            ONNXNode featureEmbedding = input.apply(ONNXOperators.GEMM, embWeight);

            // Square the output
            ONNXNode embeddingSquared = featureEmbedding.apply(ONNXOperators.POW, twoConst);

            // Square the embeddings
            ONNXNode embWeightSquared = embWeight.apply(ONNXOperators.POW, twoConst);

            // squared features * squared embeddings
            ONNXNode inputByEmbeddingSquared = inputSquared.apply(ONNXOperators.GEMM, embWeightSquared);

            // squared product subtract product of squares
            ONNXNode subtract = embeddingSquared.apply(ONNXOperators.SUB, inputByEmbeddingSquared);

            // sum over embedding dimensions
            // Divide by 2
            embeddingOutputs.add(subtract.apply(ONNXOperators.REDUCE_SUM, sumAxes)
                    .apply(ONNXOperators.DIV, twoConst));
        }

        ONNXNode concat = onnx.operation(ONNXOperators.CONCAT, embeddingOutputs, "fm_concat", Collections.singletonMap("axis", 1));

        return onnxOutput(gemm.apply(ONNXOperators.ADD, concat));
    }

    /**
     * Exports this {@link org.tribuo.Model} as an ONNX protobuf.
     * @param domain A reverse-DNS name to namespace the model (e.g., org.tribuo.classification.sgd.linear).
     * @param modelVersion A version number for this model.
     * @return The ONNX ModelProto representing this Tribuo Model.
     */
    public OnnxMl.ModelProto exportONNXModel(String domain, long modelVersion) {
        ONNXContext onnx = new ONNXContext();
        onnx.setName(onnxModelName());
        ONNXPlaceholder input = onnx.floatInput("input", featureIDMap.size());
        ONNXPlaceholder output = onnx.floatOutput("output", outputIDInfo.size());
        writeONNXGraph(input).assignTo(output);
        return ONNXExportable.buildModel(onnx, domain, modelVersion, this);
    }

}
