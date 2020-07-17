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

package org.tribuo.classification.dtree;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.Dataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.dtree.impl.ClassifierTrainingNode;
import org.tribuo.classification.dtree.impurity.GiniIndex;
import org.tribuo.classification.dtree.impurity.LabelImpurity;
import org.tribuo.common.tree.AbstractCARTTrainer;
import org.tribuo.common.tree.AbstractTrainingNode;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

/**
 * A {@link org.tribuo.Trainer} that uses an approximation of the CART algorithm to build a decision tree.
 * <p>
 * See:
 * <pre>
 * J. Friedman, T. Hastie, &amp; R. Tibshirani.
 * "The Elements of Statistical Learning"
 * Springer 2001. <a href="http://web.stanford.edu/~hastie/ElemStatLearn/">PDF</a>
 * </pre>
 */
public class CARTClassificationTrainer extends AbstractCARTTrainer<Label> {

    /**
     * Impurity measure used to determine split quality.
     */
    @Config(description = "The impurity measure used to determine split quality.")
    private LabelImpurity impurity = new GiniIndex();

    /**
     * Creates a CART Trainer.
     *
     * @param maxDepth The maximum depth of the tree.
     * @param minChildWeight The minimum node weight to consider it for a split.
     * @param fractionFeaturesInSplit The fraction of features available in each split.
     * @param impurity Impurity measure to determine split quality. See {@link LabelImpurity}.
     * @param seed The RNG seed.
     */
    public CARTClassificationTrainer(
            int maxDepth,
            float minChildWeight,
            float fractionFeaturesInSplit,
            LabelImpurity impurity,
            long seed
    ) {
        super(maxDepth, minChildWeight, fractionFeaturesInSplit, seed);
        this.impurity = impurity;
        postConfig();
    }

    /**
     * Creates a CART Trainer. Sets the impurity to the {@link GiniIndex}.
     */
    public CARTClassificationTrainer() {
        this(Integer.MAX_VALUE);
    }

    /**
     * Creates a CART trainer. Sets the impurity to the {@link GiniIndex}, uses
     * all the features, and sets the minimum number of examples in a leaf to {@link #MIN_EXAMPLES}.
     * @param maxDepth The maximum depth of the tree.
     */
    public CARTClassificationTrainer(int maxDepth) {
        this(maxDepth, MIN_EXAMPLES, 1.0f, new GiniIndex(), Trainer.DEFAULT_SEED);
    }

    /**
     * Creates a CART Trainer. Sets the impurity to the {@link GiniIndex}.
     * @param maxDepth The maximum depth of the tree.
     * @param fractionFeaturesInSplit The fraction of features available in each split.
     * @param seed The seed for the RNG.
     */
    public CARTClassificationTrainer(int maxDepth, float fractionFeaturesInSplit, long seed) {
        this(maxDepth, MIN_EXAMPLES, fractionFeaturesInSplit, new GiniIndex(), seed);
    }

    @Override
    protected AbstractTrainingNode<Label> mkTrainingNode(Dataset<Label> examples) {
        return new ClassifierTrainingNode(impurity, examples);
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("CARTClassificationTrainer(maxDepth=");
        buffer.append(maxDepth);
        buffer.append(",minChildWeight=");
        buffer.append(minChildWeight);
        buffer.append(",fractionFeaturesInSplit=");
        buffer.append(fractionFeaturesInSplit);
        buffer.append(",impurity=");
        buffer.append(impurity.toString());
        buffer.append(",seed=");
        buffer.append(seed);
        buffer.append(")");

        return buffer.toString();
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}