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
 * A trainer which produces an Extremely Randomized Tree Ensemble.
 * <p>
 * Extremely Randomized Trees are similar to Random Forests, but they add an extra element of randomness in that
 * the split points for features are also chosen randomly. As with Random Forests, feature subsampling is available at
 * each of the nodes.
 * An exception will be thrown if the inner trainer is not a decision tree trainer or if random splitting is turned off.
 * <p>
 * See:
 * <pre>
 * P. Geurts, D. Ernst, L. Wehenkel.
 * "Extremely Randomized Trees"
 * March 2006. <a href="https://link.springer.com/article/10.1007/s10994-006-6226-1">PDF</a>
 * </pre>
 *
 */
public class ExtraTreesTrainer<T extends Output<T>> extends BaggingTrainer<T> {

    private static final Logger logger = Logger.getLogger(ExtraTreesTrainer.class.getName());

    /**
     * For the configuration system.
     */
    private ExtraTreesTrainer() { }

    /**
     * Constructs an ExtraTreesTrainer with the default seed {@link org.tribuo.Trainer#DEFAULT_SEED}.
     * <p>
     * Throws {@link PropertyException} if the trainer is not set to use random split points.
     * @param trainer The tree trainer.
     * @param combiner The combining function for the ensemble.
     * @param numMembers The number of ensemble members to train.
     */
    public ExtraTreesTrainer(DecisionTreeTrainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers) {
        super(trainer,combiner,numMembers);
        postConfig();
    }

    /**
     * Constructs an ExtraTreesTrainer with the supplied seed, trainer, combining function and number of members.
     * <p>
     * Throws {@link PropertyException} if the trainer is not set to use random split points.
     * @param trainer The tree trainer.
     * @param combiner The combining function for the ensemble.
     * @param numMembers The number of ensemble members to train.
     * @param seed The RNG seed.
     */
    public ExtraTreesTrainer(DecisionTreeTrainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers, long seed) {
        super(trainer,combiner,numMembers,seed);
        postConfig();
    }

    @Override
    public void postConfig() {
        super.postConfig();
        if (!(innerTrainer instanceof DecisionTreeTrainer)) {
            throw new PropertyException("","innerTrainer","ExtraTreesTrainer requires a decision tree innerTrainer");
        }
        DecisionTreeTrainer<T> t = (DecisionTreeTrainer<T>) innerTrainer;
        if (!t.getUseRandomSplitPoints()) {
            throw new PropertyException("","innerTrainer","ExtraTreesTrainer requires that the decision tree " +
                    "innerTrainer have random split points turned on.");
        }
    }

    @Override
    protected String ensembleName() {
        return "extra-trees-ensemble";
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("ExtraTreesTrainer(");
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

