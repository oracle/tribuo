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

package org.tribuo.evaluation;

import org.tribuo.Dataset;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.dataset.DatasetView;
import org.tribuo.util.Util;

import java.util.Arrays;
import java.util.Iterator;
import java.util.SplittableRandom;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * A k-fold splitter to be used in cross-validation.
 *
 * @param <T> the type of the examples that make up the dataset to be split
 */
public class KFoldSplitter<T extends Output<T>> {

    private static final Logger logger = Logger.getLogger(KFoldSplitter.class.getName());

    protected final int nsplits;
    protected final long seed;
    protected final SplittableRandom rng;

    /**
     * Builds a k-fold splitter.
     * @param nsplits The number of folds.
     * @param randomSeed The RNG seed.
     */
    public KFoldSplitter(int nsplits, long randomSeed) {
        if (nsplits < 2) {
            throw new IllegalArgumentException("nsplits must be at least 2");
        }
        this.nsplits = nsplits;
        this.seed = randomSeed;
        this.rng = new SplittableRandom(randomSeed);
    }

    /**
     * Builds a k-fold splitter using {@link org.tribuo.Trainer#DEFAULT_SEED} as the seed.
     * @param nsplits The number of folds.
     */
    public KFoldSplitter(int nsplits) {
        this(nsplits, Trainer.DEFAULT_SEED);
    }

    /**
     * Splits a dataset into k consecutive folds; for each fold, the remaining k-1 folds form the training set.
     * <p>
     * Note: the first {@code nsamples % nsplits} folds have size {@code nsamples/nsplits + 1} and the remaining have
     * size {@code nsamples/nsplits}, where {@code nsamples = dataset.size()}.
     * @param dataset The full dataset
     * @param shuffle Whether or not shuffle the dataset before generating folds
     * @return An iterator over TrainTestFolds
     */
    public Iterator<TrainTestFold<T>> split(Dataset<T> dataset, boolean shuffle) {
        int nsamples = dataset.size();
        if (nsamples == 0) {
            throw new IllegalArgumentException("empty input data");
        }
        if (nsplits > nsamples) {
            throw new IllegalArgumentException("cannot have nsplits > nsamples");
        }
        int[] indices;
        if (shuffle) {
            indices = Util.randperm(nsamples,rng);
        } else {
            indices = IntStream.range(0, nsamples).toArray();
        }
        int[] foldSizes = new int[nsplits];
        Arrays.fill(foldSizes, nsamples/nsplits);
        for (int i = 0; i < (nsamples%nsplits); i++) {
            foldSizes[i] += 1;
        }

        return new Iterator<TrainTestFold<T>>() {
            int foldPtr = 0;
            int dataPtr = 0;

            @Override
            public boolean hasNext() {
                return foldPtr < foldSizes.length;
            }

            @Override
            public TrainTestFold<T> next() {
                int size = foldSizes[foldPtr];
                foldPtr++;
                int start = dataPtr;
                int stop = dataPtr+size;
                dataPtr = stop;
                int[] holdOut = Arrays.copyOfRange(indices, start, stop);
                int[] rest = new int[indices.length - holdOut.length];
                System.arraycopy(indices, 0, rest, 0, start);
                System.arraycopy(indices, stop, rest, start, nsamples-stop);
                return new TrainTestFold<>(
                        new DatasetView<>(dataset, rest, "TrainFold(seed="+seed+","+foldPtr+" of " + nsplits+")"),
                        new DatasetView<>(dataset, holdOut, "TestFold(seed="+seed+","+foldPtr+" of " + nsplits+")" )
                );
            }
        };
    }

    /**
     * Stores a train/test split for a dataset.
     * <p>
     * Will be a record one day.
     * @see KFoldSplitter#split
     *
     * @param <T> the type of the examples that make up the data we've split.
     */
    public static class TrainTestFold<T extends Output<T>> {
        /**
         * The training fold.
         */
        public final DatasetView<T> train;
        /**
         * The testing fold.
         */
        public final DatasetView<T> test;

        /**
         * Constructs a train test fold.
         * @param train The training fold.
         * @param test The testing fold.
         */
        TrainTestFold(DatasetView<T> train, DatasetView<T> test) {
            this.train = train;
            this.test = test;
        }
    }
}
