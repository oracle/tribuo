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

import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;

import java.util.Arrays;

/**
 * A collection of helper methods for performing training and inference in a CRF.
 */
public final class ChainHelper {

    private ChainHelper() { }

    /**
     * Runs belief propagation on a linear chain CRF. Uses the
     * linear predictions for each token and the label transition probabilities.
     * @param scores Tuple containing the label-label transition matrix, and the per token label scores.
     * @return Tuple containing the normalising constant, the forward values and the backward values.
     */
    public static ChainBPResults beliefPropagation(ChainCliqueValues scores) {
        int numLabels = scores.transitionValues.getDimension1Size();
        DenseMatrix markovScores = scores.transitionValues;
        DenseVector[] localScores = scores.localValues;
        DenseVector[] alphas = new DenseVector[localScores.length];
        DenseVector[] betas = new DenseVector[localScores.length];
        for (int i = 0; i < localScores.length; i++) {
            alphas[i] = localScores[i].copy();
            betas[i] = new DenseVector(numLabels, Double.NEGATIVE_INFINITY);
        }
        //
        // Forward pass
        double[] tmpArray = new double[numLabels];
        for (int i = 1; i < localScores.length; i++) {
            DenseVector curAlpha = alphas[i];
            DenseVector prevAlpha = alphas[i - 1];
            for (int vi = 0; vi < numLabels; vi++) {
                for (int vj = 0; vj < numLabels; vj++) {
                    tmpArray[vj] = markovScores.get(vj,vi) + prevAlpha.get(vj);
                }
                curAlpha.add(vi,sumLogProbs(tmpArray));
            }
        }
        //
        // Backward pass
        betas[betas.length-1].fill(0.0);
        for (int i = localScores.length - 2; i >= 0; i--) {
            DenseVector curBeta = betas[i];
            DenseVector prevBeta = betas[i + 1];
            DenseVector prevLocalScore = localScores[i + 1];
            for (int vi = 0; vi < numLabels; vi++) {
                for (int vj = 0; vj < numLabels; vj++) {
                    tmpArray[vj] = markovScores.get(vi,vj) + prevBeta.get(vj) + prevLocalScore.get(vj);
                }
                curBeta.set(vi,sumLogProbs(tmpArray));
            }
        }
        double logZ = sumLogProbs(alphas[alphas.length-1]);
        return new ChainBPResults(logZ, alphas, betas, scores);
    }

    /**
     * Runs constrained belief propagation on a linear chain CRF. Uses the
     * linear predictions for each token and the label transition probabilities.
     * <p>
     * See:
     * <pre>
     * "Confidence Estimation for Information Extraction",
     * A. Culotta and A. McCallum
     * Proceedings of HLT-NAACL 2004: Short Papers, 2004.
     * </pre>
     * @param scores Tuple containing the label-label transition matrix, and the per token label scores.
     * @param constraints An array of integers, representing the label constraints. -1 signifies no constraint, otherwise it's the label id.
     * @return The normalization constant for this constrained run.
     */
    public static double constrainedBeliefPropagation(ChainCliqueValues scores, int[] constraints) {
        int numLabels = scores.transitionValues.getDimension1Size();
        DenseMatrix markovScores = scores.transitionValues;
        DenseVector[] localScores = scores.localValues;
        if (localScores.length != constraints.length) {
            throw new IllegalArgumentException("Must have the same number of constraints as tokens");
        }
        DenseVector[] alphas = new DenseVector[localScores.length];
        for (int i = 0; i < localScores.length; i++) {
            alphas[i] = localScores[i].copy();
        }
        //
        // Forward pass
        double[] tmpArray = new double[numLabels];
        for (int i = 1; i < localScores.length; i++) {
            DenseVector curAlpha = alphas[i];
            DenseVector prevAlpha = alphas[i - 1];
            for (int vi = 0; vi < numLabels; vi++) {
                if ((constraints[i] == -1) || (constraints[i] == vi)) {
                    // if unconstrained or path conforms to constraints
                    for (int vj = 0; vj < numLabels; vj++) {
                        tmpArray[vj] = markovScores.get(vj,vi) + prevAlpha.get(vj);
                    }
                    curAlpha.add(vi,sumLogProbs(tmpArray));
                } else {
                    // Path is outside constraints, set to zero as alpha is initialised with the local scores.
                    curAlpha.set(vi,Double.NEGATIVE_INFINITY);
                }
            }
        }
        return sumLogProbs(alphas[alphas.length-1]);
    }

    /**
     * Runs Viterbi on a linear chain CRF. Uses the
     * linear predictions for each token and the label transition probabilities.
     * @param scores Tuple containing the label-label transition matrix, and the per token label scores.
     * @return Tuple containing the score of the maximum path and the maximum predicted label per token.
     */
    public static ChainViterbiResults viterbi(ChainCliqueValues scores) {
        DenseMatrix markovScores = scores.transitionValues;
        DenseVector[] localScores = scores.localValues;
        int numLabels = markovScores.getDimension1Size();
        DenseVector[] costs = new DenseVector[scores.localValues.length];
        int[][] backPointers = new int[scores.localValues.length][];
        for (int i = 0; i < scores.localValues.length; i++) {
            costs[i] = new DenseVector(numLabels, Double.NEGATIVE_INFINITY);
            backPointers[i] = new int[numLabels];
            Arrays.fill(backPointers[i],-1);
        }
        costs[0].setElements(localScores[0]);
        for (int i = 1; i < scores.localValues.length; i++) {
            DenseVector curLocalScores = localScores[i];
            DenseVector curCost = costs[i];
            int[] curBackPointers = backPointers[i];
            DenseVector prevCost = costs[i - 1];
            for (int vi = 0; vi < numLabels; vi++) {
                double maxScore = Double.NEGATIVE_INFINITY;
                int maxIndex = -1;
                double curLocalScore = curLocalScores.get(vi);

                for (int vj = 0; vj < numLabels; vj++) {
                    double curScore = markovScores.get(vj, vi) + prevCost.get(vj) + curLocalScore;
                    if (curScore > maxScore) {
                        maxScore = curScore;
                        maxIndex = vj;
                    }
                }
                curCost.set(vi,maxScore);
                if (maxIndex < 0) {
                    maxIndex = 0;
                }
                curBackPointers[vi] = maxIndex;
            }
        }
        int[] mapValues = new int[scores.localValues.length];
        mapValues[mapValues.length - 1] = costs[costs.length-1].indexOfMax();
        for (int j = mapValues.length - 2; j >= 0; j--) {
            mapValues[j] = backPointers[j + 1][mapValues[j + 1]];
        }
        return new ChainViterbiResults(costs[costs.length-1].maxValue(), mapValues, scores);
    }

    /**
     * Sums the log probabilities. Must be updated in concert with {@link ChainHelper#sumLogProbs(double[])}.
     * @param input A {@link DenseVector} of log probabilities.
     * @return log sum exp input[i].
     */
    public static double sumLogProbs(DenseVector input) {
        double LOG_TOLERANCE = 30.0;

        double maxValue = input.get(0);
        int maxIdx = 0;
        for (int i = 1; i < input.size(); i++) {
            double value = input.get(i);
            if (value > maxValue) {
                maxValue = value;
                maxIdx = i;
            }
        }
        if (maxValue == Double.NEGATIVE_INFINITY) {
            return maxValue;
        }

        boolean anyAdded = false;
        double intermediate = 0.0;
        double cutoff = maxValue - LOG_TOLERANCE;
        for (int i = 0; i < input.size(); i++) {
            double value = input.get(i);
            if (value >= cutoff && i != maxIdx && !Double.isInfinite(value)) {
                anyAdded = true;
                intermediate += Math.exp(value - maxValue);
            }
        }
        if (anyAdded) {
            return maxValue + Math.log1p(intermediate);
        } else {
            return maxValue;
        }
    }

    /**
     * Sums the log probabilities. Must be updated in concert with {@link ChainHelper#sumLogProbs(DenseVector)}.
     * @param input A double array of log probabilities.
     * @return log sum exp input[i].
     */
    public static double sumLogProbs(double[] input) {
        double LOG_TOLERANCE = 30.0;

        double maxValue = input[0];
        int maxIdx = 0;
        for (int i = 1; i < input.length; i++) {
            double value = input[i];
            if (value > maxValue) {
                maxValue = value;
                maxIdx = i;
            }
        }
        if (maxValue == Double.NEGATIVE_INFINITY) {
            return maxValue;
        }

        boolean anyAdded = false;
        double intermediate = 0.0;
        double cutoff = maxValue - LOG_TOLERANCE;
        for (int i = 0; i < input.length; i++) {
            if (input[i] >= cutoff && i != maxIdx && !Double.isInfinite(input[i])) {
                anyAdded = true;
                intermediate += Math.exp(input[i] - maxValue);
            }
        }
        if (anyAdded) {
            return maxValue + Math.log1p(intermediate);
        } else {
            return maxValue;
        }
    }

    /**
     * Belief Propagation results. One day it'll be a record, but not today.
     */
    public static final class ChainBPResults {
        /**
         * The normalization constant.
         */
        public final double logZ;
        /**
         * The alpha values array from forward propagation.
         */
        public final DenseVector[] alphas;
        /**
         * The beta values array from backward propagation.
         */
        public final DenseVector[] betas;
        /**
         * The local clique scores (i.e., the linear model scores).
         */
        public final ChainCliqueValues scores;

        ChainBPResults(double logZ, DenseVector[] alphas, DenseVector[] betas, ChainCliqueValues scores) {
            this.logZ = logZ;
            this.alphas = alphas;
            this.betas = betas;
            this.scores = scores;
        }
    }

    /**
     * Clique scores within a chain. One day it'll be a record, but not today.
     */
    public static final class ChainCliqueValues {
        /**
         * The per element values.
         */
        public final DenseVector[] localValues;
        /**
         * The label-label transition matrix.
         */
        public final DenseMatrix transitionValues;

        ChainCliqueValues(DenseVector[] localValues, DenseMatrix transitionValues) {
            this.localValues = localValues;
            this.transitionValues = transitionValues;
        }
    }

    /**
     * Viterbi output from a linear chain. One day it'll be a record, but not today.
     */
    public static final class ChainViterbiResults {
        /**
         * The score for this result.
         */
        public final double mapScore;
        /**
         * The viterbi states.
         */
        public final int[] mapValues;
        /**
         * The pre-viterbi scores.
         */
        public final ChainCliqueValues scores;

        ChainViterbiResults(double mapScore, int[] mapValues, ChainCliqueValues scores) {
            this.mapScore = mapScore;
            this.mapValues = mapValues;
            this.scores = scores;
        }
    }
}
