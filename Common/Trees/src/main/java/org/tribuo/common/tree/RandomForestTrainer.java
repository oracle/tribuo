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

package org.tribuo.common.tree;

import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.tribuo.Output;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.ensemble.EnsembleCombiner;

import java.util.logging.Logger;

/**
 * A trainer which produces a random forest.
 * <p>
 * Random Forests are basically bagged trees, with feature subsampling at each of the nodes.
 * An exception will be thrown if the user does not supply a decision tree trainer with feature subsampling turned on
 * and random splitting turned off.
 * <p>
 * See:
 * <pre>
 * J. Friedman, T. Hastie, &amp; R. Tibshirani.
 * "The Elements of Statistical Learning"
 * Springer 2001. <a href="http://web.stanford.edu/~hastie/ElemStatLearn/">PDF</a>
 * </pre>
 */
public class RandomForestTrainer<T extends Output<T>> extends BaggingTrainer<T> {

    private static final Logger logger = Logger.getLogger(RandomForestTrainer.class.getName());

    /**
     * For the configuration system.
     */
    private RandomForestTrainer() { }

    /**
     * Constructs a RandomForestTrainer with the default seed {@link org.tribuo.Trainer#DEFAULT_SEED}.
     * <p>
     * Throws {@link PropertyException} if the trainer is not set to subsample the features.
     * @param trainer The tree trainer.
     * @param combiner The combining function for the ensemble.
     * @param numMembers The number of ensemble members to train.
     */
    public RandomForestTrainer(DecisionTreeTrainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers) {
        super(trainer,combiner,numMembers);
        postConfig();
    }

    /**
     * Constructs a RandomForestTrainer with the supplied seed, trainer, combining function and number of members.
     * <p>
     * Throws {@link PropertyException} if the trainer is not set to subsample the features.
     * @param trainer The tree trainer.
     * @param combiner The combining function for the ensemble.
     * @param numMembers The number of ensemble members to train.
     * @param seed The RNG seed.
     */
    public RandomForestTrainer(DecisionTreeTrainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers, long seed) {
        super(trainer,combiner,numMembers,seed);
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        super.postConfig();
        if (!(innerTrainer instanceof DecisionTreeTrainer)) {
            throw new PropertyException("","innerTrainer","RandomForestTrainer requires a decision tree innerTrainer");
        }

        DecisionTreeTrainer<T> t = (DecisionTreeTrainer<T>) innerTrainer;
        if (t.getFractionFeaturesInSplit() == 1f) {
            throw new PropertyException("","innerTrainer","RandomForestTrainer requires that the decision tree " +
                    "innerTrainer have fractional features in split.");
        }

        if (t.getUseRandomSplitPoints()) {
            throw new PropertyException("","innerTrainer","RandomForestTrainer requires that the decision tree " +
                    "use non-random splitting, but useRandomSplits was true. If you want random splits, use " +
                    "ExtraTreesTrainer instead.");
        }
    }

    @Override
    protected String ensembleName() {
        return "random-forest-ensemble";
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("RandomForestTrainer(");
        buffer.append("innerTrainer=");
        buffer.append(innerTrainer.toString());
        buffer.append(",combiner=");
        buffer.append(combiner.toString());
        buffer.append(",numMembers=");
        buffer.append(numMembers);
        buffer.append(",seed=");
        buffer.append(seed);
        buffer.append(")");

        return buffer.toString();
    }
    
}
