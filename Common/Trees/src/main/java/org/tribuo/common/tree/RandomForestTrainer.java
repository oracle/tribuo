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
 * It's up to the user to supply a decision tree trainer which has feature subsampling turned on by
 * checking {@link DecisionTreeTrainer#getFractionFeaturesInSplit()}.
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

    public RandomForestTrainer(DecisionTreeTrainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers) {
        super(trainer,combiner,numMembers);
    }

    public RandomForestTrainer(DecisionTreeTrainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers, long seed) {
        super(trainer,combiner,numMembers,seed);
    }

    @Override
    public void postConfig() {
        super.postConfig();
        if (!(innerTrainer instanceof DecisionTreeTrainer)) {
            throw new PropertyException("","innerTrainer","RandomForestTrainer requires a decision tree innerTrainer");
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
