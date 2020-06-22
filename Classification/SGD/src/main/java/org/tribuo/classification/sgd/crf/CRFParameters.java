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

package org.tribuo.classification.sgd.crf;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.util.HeapMerger;
import org.tribuo.math.util.Merger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A {@link Parameters} for training a CRF using SGD.
 */
public class CRFParameters implements Parameters, Serializable {
    private static final long serialVersionUID = 1L;

    private final int numLabels;
    private final int numFeatures;

    private static final Merger merger = new HeapMerger();

    /**
     * This variable is an array with 3 elements corresponding to the three weight matrices.
     */
    private Tensor[] weights;

    private DenseVector biases; //weights[0];
    private DenseMatrix featureLabelWeights; //weights[1];
    private DenseMatrix labelLabelWeights; //weights[2];

    CRFParameters(int numFeatures, int numLabels) {
        this.biases = new DenseVector(numLabels);
        this.featureLabelWeights = new DenseMatrix(numLabels,numFeatures);
        this.labelLabelWeights = new DenseMatrix(numLabels,numLabels);
        this.weights = new Tensor[3];
        weights[0] = biases;
        weights[1] = featureLabelWeights;
        weights[2] = labelLabelWeights;
        this.numLabels = numLabels;
        this.numFeatures = numFeatures;
    }

    public DenseVector getFeatureWeights(int id) {
        return featureLabelWeights.getColumn(id);
    }

    public double getBias(int id) {
        return biases.get(id);
    }

    public double getWeight(int labelID, int featureID) {
        return featureLabelWeights.get(labelID, featureID);
    }

    /**
     * Generate the local scores (i.e. the linear classifier for each token).
     * @param features An array of {@link SparseVector}s, one per token.
     * @return An array of DenseVectors representing the scores per label for each token.
     */
    public DenseVector[] getLocalScores(SparseVector[] features) {
        DenseVector[] localScores = new DenseVector[features.length];
        for (int i = 0; i < features.length; i++) {
            DenseVector scores = featureLabelWeights.leftMultiply(features[i]);
            scores.intersectAndAddInPlace(biases);
            localScores[i] = scores;
        }
        return localScores;
    }

    /**
     * Generates the local scores and tuples them with the label - label transition weights.
     * @param features The per token {@link SparseVector} of features.
     * @return A tuple containing the array of {@link DenseVector} scores and the label - label transition weights.
     */
    public ChainHelper.ChainCliqueValues getCliqueValues(SparseVector[] features) {
        DenseVector[] localScores = getLocalScores(features);
        return new ChainHelper.ChainCliqueValues(localScores, labelLabelWeights);
    }

    /**
     * Generate a prediction using Viterbi.
     * @param features The per token {@link SparseVector} of features.
     * @return An int array giving the predicted label per token.
     */
    public int[] predict(SparseVector[] features) {
        ChainHelper.ChainViterbiResults result = ChainHelper.viterbi(getCliqueValues(features));
        return result.mapValues;
    }

    /**
     * Generate a prediction using Belief Propagation.
     * @param features The per token {@link SparseVector} of features.
     * @return A {@link DenseVector} per token containing the marginal distribution over labels.
     */
    public DenseVector[] predictMarginals(SparseVector[] features) {
        ChainHelper.ChainBPResults result = ChainHelper.beliefPropagation(getCliqueValues(features));
        DenseVector[] marginals = new DenseVector[features.length];
        for (int i = 0; i < features.length; i++) {
            marginals[i] = result.alphas[i].add(result.betas[i]);
            marginals[i].expNormalize(result.logZ);
        }
        return marginals;
    }

    /**
     * This predicts per chunk confidence using the constrained forward backward algorithm from
     * Culotta and McCallum 2004.
     * <p>
     * Runs one pass of BP to get the normalizing constant, and then a further chunks.size() passes
     * of constrained forward backward.
     * @param features The per token {@link SparseVector} of features.
     * @param chunks A list of extracted chunks to pin constrain the labels to.
     * @return A list containing the confidence value for each chunk.
     */
    public List<Double> predictConfidenceUsingCBP(SparseVector[] features, List<Chunk> chunks) {
        ChainHelper.ChainCliqueValues cliqueValues = getCliqueValues(features);
        ChainHelper.ChainBPResults bpResult = ChainHelper.beliefPropagation(cliqueValues);
        double bpLogZ = bpResult.logZ;

        int[] constraints = new int[features.length];

        List<Double> output = new ArrayList<>();
        for (Chunk chunk : chunks) {
            Arrays.fill(constraints,-1);
            chunk.unpack(constraints);
            double chunkScore = ChainHelper.constrainedBeliefPropagation(cliqueValues,constraints);
            output.add(Math.exp(chunkScore - bpLogZ));
        }

        return output;
    }

    /**
     * Generates predictions based on the input features and labels, then scores those predictions to
     * produce a loss for the example and a gradient update.
     * @param features The per token {@link SparseVector} of features.
     * @param labels The per token ground truth labels.
     * @return A {@link Pair} containing the loss for this example and the associated gradient.
     */
    public Pair<Double, Tensor[]> valueAndGradient(SparseVector[] features, int[] labels) {
        ChainHelper.ChainCliqueValues scores = getCliqueValues(features);
        // Infer the marginal distribution over labels for each token.
        ChainHelper.ChainBPResults bpResults = ChainHelper.beliefPropagation(scores);
        double logZ = bpResults.logZ;
        DenseVector[] alphas = bpResults.alphas;
        DenseVector[] betas = bpResults.betas;

        //Calculate the gradients for the parameters.
        Tensor[] gradient = new Tensor[3];
        DenseSparseMatrix[] featureGradients = new DenseSparseMatrix[features.length];
        gradient[0] = new DenseVector(biases.size());
        Matrix transGradient = new DenseMatrix(numLabels, numLabels);
        gradient[2] = transGradient;
        double score = -logZ;
        for (int i = 0; i < features.length; i++) {
            int curLabel = labels[i];
            // Increment the loss based on the score for the true label.
            DenseVector curLocalScores = scores.localValues[i];
            score += curLocalScores.get(curLabel);
            // Generate the predicted local marginal from the BP run.
            DenseVector curAlpha = alphas[i];
            DenseVector curBeta = betas[i];
            DenseVector localMarginal = curAlpha.add(curBeta);
            // Generate the gradient for the biases based on the true label and predicted label.
            localMarginal.expNormalize(logZ);
            localMarginal.scaleInPlace(-1.0);
            localMarginal.add(curLabel,1.0);
            gradient[0].intersectAndAddInPlace(localMarginal);
            // Generate the gradient for the feature - label weights
            featureGradients[i] = (DenseSparseMatrix) localMarginal.outer(features[i]);
            // If the sequence has more than one token generate the gradient for the label - label transitions.
            if (i >= 1) {
                DenseVector prevAlpha = alphas[i - 1];
                for (int ii = 0; ii < numLabels; ii++) {
                    for (int jj = 0; jj < numLabels; jj++) {
                        double update = -Math.exp(prevAlpha.get(ii) + labelLabelWeights.get(ii,jj) + curBeta.get(jj) + curLocalScores.get(jj) - logZ);
                        transGradient.add(ii, jj, update);
                    }
                }
                int prevLabel = labels[i-1];
                // Increment the loss based on the transition from the previous predicted label to the true label.
                score += (labelLabelWeights.get(prevLabel,curLabel));
                transGradient.add(prevLabel, curLabel, 1.0);
            }
        }
        // Merge together all the sparse feature - label gradients.
        gradient[1] = merger.merge(featureGradients);

        return new Pair<>(score,gradient);
    }

    /**
     * Returns a 3 element {@link Tensor} array.
     *
     * The first element is a {@link DenseVector} of label biases.
     * The second element is a {@link DenseMatrix} of feature-label weights.
     * The third element is a {@link DenseMatrix} of label-label transition weights.
     * @return A {@link Tensor} array.
     */
    @Override
    public Tensor[] getEmptyCopy() {
        Tensor[] output = new Tensor[3];
        output[0] = new DenseVector(biases.size());
        output[1] = new DenseMatrix(featureLabelWeights.getDimension1Size(),featureLabelWeights.getDimension2Size());
        output[2] = new DenseMatrix(labelLabelWeights.getDimension1Size(),labelLabelWeights.getDimension2Size());
        return output;
    }

    @Override
    public Tensor[] get() {
        return weights;
    }

    @Override
    public void set(Tensor[] newWeights) {
        if (newWeights.length == weights.length) {
            weights = newWeights;
            biases = (DenseVector) weights[0];
            featureLabelWeights = (DenseMatrix) weights[1];
            labelLabelWeights = (DenseMatrix) weights[2];
        }
    }

    @Override
    public void update(Tensor[] gradients) {
        for (int i = 0; i < gradients.length; i++) {
            weights[i].intersectAndAddInPlace(gradients[i]);
        }
    }

    @Override
    public Tensor[] merge(Tensor[][] gradients, int size) {
        DenseVector biasUpdate = new DenseVector(biases.size());
        DenseSparseMatrix[] updates = new DenseSparseMatrix[size];
        DenseMatrix labelLabelUpdate = new DenseMatrix(labelLabelWeights.getDimension1Size(),labelLabelWeights.getDimension2Size());
        for (int j = 0; j < updates.length; j++) {
            biasUpdate.intersectAndAddInPlace(gradients[j][0]);
            updates[j] = (DenseSparseMatrix) gradients[j][1];
            labelLabelUpdate.intersectAndAddInPlace(gradients[j][2]);
        }

        DenseSparseMatrix featureLabelUpdate = merger.merge(updates);

        return new Tensor[]{biasUpdate,featureLabelUpdate,labelLabelUpdate};
    }
}
