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

package org.tribuo.common.nearest;

import com.oracle.labs.mlrg.olcut.config.ArgumentException;
import com.oracle.labs.mlrg.olcut.config.Option;
import org.tribuo.classification.ClassificationOptions;
import org.tribuo.classification.Label;
import org.tribuo.classification.ensemble.FullyWeightedVotingCombiner;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.common.nearest.KNNModel.Backend;
import org.tribuo.common.nearest.KNNTrainer.Distance;
import org.tribuo.ensemble.EnsembleCombiner;

/**
 * CLI Options for training a k-nearest neighbour predictor.
 */
public class KNNClassifierOptions implements ClassificationOptions<KNNTrainer<Label>> {

    /**
     * The type of combination function.
     */
    public enum EnsembleCombinerType {
        /**
         * Use a {@link VotingCombiner}.
         */
        VOTING,
        /**
         * Use a {@link FullyWeightedVotingCombiner}.
         */
        FULLY_WEIGHTED_VOTING
    }

    /**
     * K nearest neighbours to use. Defaults to 1.
     */
    @Option(longName = "knn-k", usage = "K nearest neighbours to use.")
    public int knnK = 1;
    /**
     * Number of threads to use. Defaults to 1.
     */
    @Option(longName = "knn-num-threads", usage = "Number of threads to use.")
    public int knnNumThreads = 1;
    /**
     * Distance metric to use. Defaults to L2.
     */
    @Option(longName = "knn-distance", usage = "Distance metric to use.")
    public Distance knnDistance = Distance.L2;
    /**
     * Parallel backend to use.
     */
    @Option(longName = "knn-backend", usage = "Parallel backend to use.")
    public Backend knnBackend = Backend.STREAMS;
    /**
     * Parallel backend to use.
     */
    @Option(longName = "knn-voting", usage = "Parallel backend to use.")
    public EnsembleCombinerType knnEnsembleCombiner = EnsembleCombinerType.VOTING;

    @Override
    public String getOptionsDescription() {
        return "Options for parameterising a LibLinear classification trainer.";
    }

    private EnsembleCombiner<Label> getEnsembleCombiner() {
        switch (knnEnsembleCombiner) {
            case VOTING:
                return new VotingCombiner();
            case FULLY_WEIGHTED_VOTING:
                return new FullyWeightedVotingCombiner();
            default:
                throw new ArgumentException("ensemble combiner", "Unknown ensemble combiner " + knnEnsembleCombiner);
        }
    }

    @Override
    public KNNTrainer<Label> getTrainer() {
        return new KNNTrainer<>(knnK, knnDistance, knnNumThreads, getEnsembleCombiner(), knnBackend);
    }
}
