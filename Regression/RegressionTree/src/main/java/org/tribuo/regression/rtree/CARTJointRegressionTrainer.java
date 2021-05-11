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

package org.tribuo.regression.rtree;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.Dataset;
import org.tribuo.Trainer;
import org.tribuo.common.tree.AbstractCARTTrainer;
import org.tribuo.common.tree.AbstractTrainingNode;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.rtree.impl.JointRegressorTrainingNode;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;
import org.tribuo.regression.rtree.impurity.RegressorImpurity;

/**
 * A {@link org.tribuo.Trainer} that uses an approximation of the CART algorithm to build a decision tree.
 * <p>
 * Builds a single tree for all the regression dimensions.
 * <p>
 * See:
 * <pre>
 * J. Friedman, T. Hastie, &amp; R. Tibshirani.
 * "The Elements of Statistical Learning"
 * Springer 2001. <a href="http://web.stanford.edu/~hastie/ElemStatLearn/">PDF</a>
 * </pre>
 */
public class CARTJointRegressionTrainer extends AbstractCARTTrainer<Regressor> {

    /**
     * Impurity measure used to determine split quality.
     */
    @Config(description="The regression impurity to use.")
    private RegressorImpurity impurity = new MeanSquaredError();

    /**
     * Normalizes the output of each leaf so it sums to one (i.e., is a probability distribution).
     */
    @Config(description="Normalize the output of each leaf so it sums to one.")
    private boolean normalize = false;

    /**
     * Creates a CART Trainer.
     *
     * @param maxDepth maxDepth The maximum depth of the tree.
     * @param minChildWeight minChildWeight The minimum node weight to consider it for a split.
     * @param minImpurityDecrease The minimum decrease in impurity necessary to split a node.
     * @param fractionFeaturesInSplit fractionFeaturesInSplit The fraction of features available in each split.
     * @param useRandomSplitPoints Whether to choose split points for features at random.
     * @param impurity impurity The impurity function to use to determine split quality.
     * @param normalize Normalize the leaves so each output sums to one.
     * @param seed The seed to use for the RNG.
     */
    public CARTJointRegressionTrainer(
            int maxDepth,
            float minChildWeight,
            float minImpurityDecrease,
            float fractionFeaturesInSplit,
            boolean useRandomSplitPoints,
            RegressorImpurity impurity,
            boolean normalize,
            long seed
    ) {
        super(maxDepth, minChildWeight, minImpurityDecrease, fractionFeaturesInSplit, useRandomSplitPoints, seed);
        this.impurity = impurity;
        this.normalize = normalize;
        postConfig();
    }

    /**
     * Creates a CART Trainer.
     * <p>
     * Computes the exact split point.
     * @param maxDepth maxDepth The maximum depth of the tree.
     * @param minChildWeight minChildWeight The minimum node weight to consider it for a split.
     * @param minImpurityDecrease The minimum decrease in impurity necessary to split a node.
     * @param fractionFeaturesInSplit fractionFeaturesInSplit The fraction of features available in each split.
     * @param impurity impurity The impurity function to use to determine split quality.
     * @param normalize Normalize the leaves so each output sums to one.
     * @param seed The seed to use for the RNG.
     */
    public CARTJointRegressionTrainer(
            int maxDepth,
            float minChildWeight,
            float minImpurityDecrease,
            float fractionFeaturesInSplit,
            RegressorImpurity impurity,
            boolean normalize,
            long seed
    ) {
        this(maxDepth, minChildWeight, minImpurityDecrease, fractionFeaturesInSplit, false, impurity, normalize, seed);
    }

    /**
     * Creates a CART Trainer.
     * <p>
     * Sets the impurity to the {@link MeanSquaredError}, computes an arbitrary depth
     * tree with exact split points using all the features, and does not normalize the outputs.
     */
    public CARTJointRegressionTrainer() {
        this(Integer.MAX_VALUE, MIN_EXAMPLES, 0.0f, 1.0f, false, new MeanSquaredError(), false, Trainer.DEFAULT_SEED);
    }

    /**
     * Creates a CART Trainer.
     * <p>
     * Sets the impurity to the {@link MeanSquaredError}, computes the exact split
     * points using all the features, and does not normalize the outputs.
     * @param maxDepth The maximum depth of the tree.
     */
    public CARTJointRegressionTrainer(int maxDepth) {
        this(maxDepth, MIN_EXAMPLES, 0.0f, 1.0f, false, new MeanSquaredError(), false, Trainer.DEFAULT_SEED);
    }

    /**
     * Creates a CART Trainer. Sets the impurity to the {@link MeanSquaredError}.
     * @param maxDepth The maximum depth of the tree.
     * @param normalize Normalises the leaves so each leaf has a distribution which sums to 1.0.
     */
    public CARTJointRegressionTrainer(int maxDepth, boolean normalize) {
        this(maxDepth, MIN_EXAMPLES, 0.0f, 1.0f, false, new MeanSquaredError(), normalize, Trainer.DEFAULT_SEED);
    }

    @Override
    protected AbstractTrainingNode<Regressor> mkTrainingNode(Dataset<Regressor> examples,
                                                             AbstractTrainingNode.LeafDeterminer leafDeterminer) {
        return new JointRegressorTrainingNode(impurity, examples, normalize, leafDeterminer);
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("CARTJointRegressionTrainer(maxDepth=");
        buffer.append(maxDepth);
        buffer.append(",minChildWeight=");
        buffer.append(minChildWeight);
        buffer.append(",minImpurityDecrease=");
        buffer.append(minImpurityDecrease);
        buffer.append(",fractionFeaturesInSplit=");
        buffer.append(fractionFeaturesInSplit);
        buffer.append(",useRandomSplitPoints=");
        buffer.append(useRandomSplitPoints);
        buffer.append(",impurity=");
        buffer.append(impurity.toString());
        buffer.append(",normalize=");
        buffer.append(normalize);
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