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

package org.tribuo.classification.ensemble;

import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.common.tree.DecisionTreeTrainer;
import org.tribuo.common.tree.RandomForestTrainer;
import org.tribuo.ensemble.BaggingTrainer;

import java.util.logging.Logger;

/**
 * Options for building a classification ensemble.
 */
public class ClassificationEnsembleOptions implements Options {
    private static final Logger logger = Logger.getLogger(ClassificationEnsembleOptions.class.getName());
    public enum EnsembleType { ADABOOST, BAGGING, RF }

    @Option(longName="ensemble-type",usage="Ensemble method, options are {ADABOOST, BAGGING, RF}.")
    public EnsembleType type = EnsembleType.BAGGING;
    @Option(longName="ensemble-size",usage="Number of base learners in the ensemble.")
    public int ensembleSize = -1;
    @Option(longName="ensemble-seed",usage="RNG seed.")
    public long seed = Trainer.DEFAULT_SEED;


    public Trainer<Label> wrapTrainer(Trainer<Label> trainer) {
        if ((ensembleSize > 0) && (type != null)) {
            switch (type) {
                case ADABOOST:
                    logger.info("Using Adaboost with " + ensembleSize + " members.");
                    return new AdaBoostTrainer(trainer,ensembleSize,seed);
                case BAGGING:
                    logger.info("Using Bagging with " + ensembleSize + " members.");
                    return new BaggingTrainer<>(trainer,new VotingCombiner(),ensembleSize,seed);
                case RF:
                    if (trainer instanceof DecisionTreeTrainer) {
                        logger.info("Using Random Forests with " + ensembleSize + " members.");
                        return new RandomForestTrainer<>((DecisionTreeTrainer<Label>)trainer,new VotingCombiner(),ensembleSize,seed);
                    } else {
                        throw new IllegalArgumentException("RandomForestTrainer requires a DecisionTreeTrainer");
                    }
                default:
                    throw new IllegalArgumentException("Unknown ensemble type :" + type);
            }
        } else {
            return trainer;
        }
    }
}
