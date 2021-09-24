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

package org.tribuo.classification.sgd;

import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;

import java.util.SplittableRandom;

/**
 * SGD utilities. Currently stores methods for shuffling examples and their associated labels and weights.
 */
public class Util {
    /**
     * In place shuffle of the features, labels and weights.
     * @deprecated In favour of {@link org.tribuo.common.sgd.AbstractLinearSGDTrainer#shuffleInPlace}.
     * @param features Input features.
     * @param labels Input labels.
     * @param weights Input weights.
     * @param rng SplittableRandom number generator.
     */
    @Deprecated
    public static void shuffleInPlace(SparseVector[] features, int[] labels, double[] weights, SplittableRandom rng) {
        int size = features.length;
        // Shuffle array
        for (int i = size; i > 1; i--) {
            int j = rng.nextInt(i);
            //swap features
            SparseVector tmpFeature = features[i-1];
            features[i-1] = features[j];
            features[j] = tmpFeature;
            //swap labels
            int tmpLabel = labels[i-1];
            labels[i-1] = labels[j];
            labels[j] = tmpLabel;
            //swap weights
            double tmpWeight = weights[i-1];
            weights[i-1] = weights[j];
            weights[j] = tmpWeight;
        }
    }

    /**
     * In place shuffle of the features, labels, weights and indices.
     * @param features Input features.
     * @param labels Input labels.
     * @param weights Input weights.
     * @param indices Input indices.
     * @param rng SplittableRandom number generator.
     */
    public static void shuffleInPlace(SparseVector[] features, int[] labels, double[] weights, int[] indices, SplittableRandom rng) {
        int size = features.length;
        // Shuffle array
        for (int i = size; i > 1; i--) {
            int j = rng.nextInt(i);
            //swap features
            SparseVector tmpFeature = features[i-1];
            features[i-1] = features[j];
            features[j] = tmpFeature;
            //swap labels
            int tmpLabel = labels[i-1];
            labels[i-1] = labels[j];
            labels[j] = tmpLabel;
            //swap weights
            double tmpWeight = weights[i-1];
            weights[i-1] = weights[j];
            weights[j] = tmpWeight;
            //swap indices
            int tmpIndex = indices[i-1];
            indices[i-1] = indices[j];
            indices[j] = tmpIndex;
        }
    }

    /**
     * Shuffles the features, labels and weights returning a tuple of the shuffled inputs.
     * @param features Input features.
     * @param labels Input labels.
     * @param weights Input weights.
     * @param rng SplittableRandom number generator.
     * @return A tuple of shuffled features, labels and weights.
     */
    public static ExampleArray shuffle(SparseVector[] features, int[] labels, double[] weights, SplittableRandom rng) {
        int size = features.length;
        SparseVector[] newFeatures = new SparseVector[size];
        int[] newLabels = new int[size];
        double[] newWeights = new double[size];
        for (int i = 0; i < newFeatures.length; i++) {
            newFeatures[i] = features[i];
            newLabels[i] = labels[i];
            newWeights[i] = weights[i];
        }
        // Shuffle array
        for (int i = size; i > 1; i--) {
            int j = rng.nextInt(i);
            //swap features
            SparseVector tmpFeature = newFeatures[i-1];
            newFeatures[i-1] = newFeatures[j];
            newFeatures[j] = tmpFeature;
            //swap labels
            int tmpLabel = newLabels[i-1];
            newLabels[i-1] = newLabels[j];
            newLabels[j] = tmpLabel;
            //swap weights
            double tmpWeight = newWeights[i-1];
            newWeights[i-1] = newWeights[j];
            newWeights[j] = tmpWeight;
        }
        return new ExampleArray(newFeatures,newLabels,newWeights);
    }

    /**
     * A nominal tuple. One day it'll be a record, but not today.
     */
    public static class ExampleArray {
        /**
         * The examples encoded as sparse vectors.
         */
        public final SparseVector[] features;
        /**
         * The label indices.
         */
        public final int[] labels;
        /**
         * The example weights.
         */
        public final double[] weights;

        /**
         * Constructs an example array.
         * @param features The examples' features.
         * @param labels The label indices.
         * @param weights The example weights.
         */
        public ExampleArray(SparseVector[] features, int[] labels, double[] weights) {
            this.features = features;
            this.labels = labels;
            this.weights = weights;
        }
    }

    /**
     * In place shuffle used for sequence problems.
     * @param features Input features.
     * @param labels Input labels.
     * @param weights Input weights.
     * @param rng SplittableRandom number generator.
     */
    public static void shuffleInPlace(SGDVector[][] features, int[][] labels, double[] weights, SplittableRandom rng) {
        int size = features.length;
        // Shuffle array
        for (int i = size; i > 1; i--) {
            int j = rng.nextInt(i);
            //swap features
            SGDVector[] tmpFeature = features[i-1];
            features[i-1] = features[j];
            features[j] = tmpFeature;
            //swap labels
            int[] tmpLabel = labels[i-1];
            labels[i-1] = labels[j];
            labels[j] = tmpLabel;
            //swap weights
            double tmpWeight = weights[i-1];
            weights[i-1] = weights[j];
            weights[j] = tmpWeight;
        }
    }

    /**
     * Shuffles a sequence of features, labels and weights, returning a tuple of the shuffled values.
     * @param features Input features.
     * @param labels Input labels.
     * @param weights Input weights.
     * @param rng SplittableRandom number generator.
     * @return A tuple of shuffled features, labels and weights.
     */
    public static SequenceExampleArray shuffle(SGDVector[][] features, int[][] labels, double[] weights, SplittableRandom rng) {
        int size = features.length;
        SGDVector[][] newFeatures = new SGDVector[size][];
        int[][] newLabels = new int[size][];
        double[] newWeights = new double[size];
        for (int i = 0; i < newFeatures.length; i++) {
            newFeatures[i] = features[i];
            newLabels[i] = labels[i];
            newWeights[i] = weights[i];
        }
        // Shuffle array
        for (int i = size; i > 1; i--) {
            int j = rng.nextInt(i);
            //swap features
            SGDVector[] tmpFeature = newFeatures[i-1];
            newFeatures[i-1] = newFeatures[j];
            newFeatures[j] = tmpFeature;
            //swap labels
            int[] tmpLabel = newLabels[i-1];
            newLabels[i-1] = newLabels[j];
            newLabels[j] = tmpLabel;
            //swap weights
            double tmpWeight = newWeights[i-1];
            newWeights[i-1] = newWeights[j];
            newWeights[j] = tmpWeight;
        }
        return new SequenceExampleArray(newFeatures,newLabels,newWeights);
    }

    /**
     * A nominal tuple. One day it'll be a record, but not today.
     */
    public static class SequenceExampleArray {
        /**
         * The array of sequence example features.
         */
        public final SGDVector[][] features;
        /**
         * The sequence example label indices.
         */
        public final int[][] labels;
        /**
         * The sequence example weights.
         */
        public final double[] weights;

        SequenceExampleArray(SGDVector[][] features, int[][] labels, double[] weights) {
            this.features = features;
            this.labels = labels;
            this.weights = weights;
        }
    }
}
