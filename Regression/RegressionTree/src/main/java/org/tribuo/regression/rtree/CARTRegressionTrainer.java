/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.common.tree.AbstractCARTTrainer;
import org.tribuo.common.tree.AbstractTrainingNode;
import org.tribuo.common.tree.Node;
import org.tribuo.common.tree.TreeModel;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.rtree.impl.RegressorTrainingNode;
import org.tribuo.regression.rtree.impl.RegressorTrainingNode.InvertedData;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;
import org.tribuo.regression.rtree.impurity.RegressorImpurity;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * A {@link org.tribuo.Trainer} that uses an approximation of the CART algorithm to build a decision tree.
 * Trains an independent tree for each output dimension.
 * <p>
 * See:
 * <pre>
 * J. Friedman, T. Hastie, &amp; R. Tibshirani.
 * "The Elements of Statistical Learning"
 * Springer 2001. <a href="http://web.stanford.edu/~hastie/ElemStatLearn/">PDF</a>
 * </pre>
 */
public final class CARTRegressionTrainer extends AbstractCARTTrainer<Regressor> {

    /**
     * Impurity measure used to determine split quality.
     */
    @Config(description="Regression impurity measure used to determine split quality.")
    private RegressorImpurity impurity = new MeanSquaredError();

    /**
     * Creates a CART Trainer.
     *
     * @param maxDepth maxDepth The maximum depth of the tree.
     * @param minChildWeight minChildWeight The minimum node weight to consider it for a split.
     * @param minImpurityDecrease The minimum decrease in impurity necessary to split a node.
     * @param fractionFeaturesInSplit fractionFeaturesInSplit The fraction of features available in each split.
     * @param useRandomSplitPoints Whether to choose split points for features at random.
     * @param impurity impurity The impurity function to use to determine split quality.
     * @param seed The RNG seed.
     */
    public CARTRegressionTrainer(
            int maxDepth,
            float minChildWeight,
            float minImpurityDecrease,
            float fractionFeaturesInSplit,
            boolean useRandomSplitPoints,
            RegressorImpurity impurity,
            long seed
    ) {
        super(maxDepth, minChildWeight, minImpurityDecrease, fractionFeaturesInSplit, useRandomSplitPoints, seed);
        this.impurity = impurity;
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
     * @param seed The RNG seed.
     */
    public CARTRegressionTrainer(
            int maxDepth,
            float minChildWeight,
            float minImpurityDecrease,
            float fractionFeaturesInSplit,
            RegressorImpurity impurity,
            long seed
    ) {
        this(maxDepth,minChildWeight,minImpurityDecrease,fractionFeaturesInSplit,false,impurity,seed);
    }

    /**
     * Creates a CART trainer.
     * <p>
     * Sets the impurity to the {@link MeanSquaredError}, uses
     * all the features, computes the exact split point, and
     * sets the minimum number of examples in a leaf to {@link #MIN_EXAMPLES}.
     */
    public CARTRegressionTrainer() {
        this(Integer.MAX_VALUE);
    }

    /**
     * Creates a CART trainer.
     * <p>
     * Sets the impurity to the {@link MeanSquaredError}, uses
     * all the features, computes the exact split point and sets
     * the minimum number of examples in a leaf to {@link #MIN_EXAMPLES}.
     * @param maxDepth The maximum depth of the tree.
     */
    public CARTRegressionTrainer(int maxDepth) {
        this(maxDepth, MIN_EXAMPLES, 0.0f, 1.0f, false, new MeanSquaredError(), Trainer.DEFAULT_SEED);
    }

    @Override
    protected AbstractTrainingNode<Regressor> mkTrainingNode(Dataset<Regressor> examples,
                                                             AbstractTrainingNode.LeafDeterminer leafDeterminer) {
        throw new IllegalStateException("Shouldn't reach here.");
    }

    @Override
    public TreeModel<Regressor> train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance) {
        return train(examples, runProvenance, INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public TreeModel<Regressor> train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // Creates a new RNG, adds one to the invocation count.
        SplittableRandom localRNG;
        TrainerProvenance trainerProvenance;
        synchronized(this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }

        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<Regressor> outputIDInfo = examples.getOutputIDInfo();
        Set<Regressor> domain = outputIDInfo.getDomain();

        int numFeaturesInSplit = Math.min(Math.round(fractionFeaturesInSplit * featureIDMap.size()),featureIDMap.size());
        int[] indices;
        int[] originalIndices = new int[featureIDMap.size()];
        for (int i = 0; i < originalIndices.length; i++) {
            originalIndices[i] = i;
        }
        if (numFeaturesInSplit != featureIDMap.size()) {
            indices = new int[numFeaturesInSplit];
        } else {
            indices = originalIndices;
        }

        float weightSum = 0.0f;
        for (Example<Regressor> e : examples) {
            weightSum += e.getWeight();
        }
        float scaledMinImpurityDecrease = getMinImpurityDecrease() * weightSum;
        AbstractTrainingNode.LeafDeterminer leafDeterminer = new AbstractTrainingNode.LeafDeterminer(maxDepth,
                minChildWeight, scaledMinImpurityDecrease);

        InvertedData data = RegressorTrainingNode.invertData(examples);

        Map<String, Node<Regressor>> nodeMap = new HashMap<>();
        for (Regressor r : domain) {
            String dimName = r.getNames()[0];
            int dimIdx = outputIDInfo.getID(r);

            AbstractTrainingNode<Regressor> root = new RegressorTrainingNode(impurity,data,dimIdx,dimName,
                    examples.size(),featureIDMap,outputIDInfo, leafDeterminer);
            Deque<AbstractTrainingNode<Regressor>> queue = new ArrayDeque<>();
            queue.add(root);

            while (!queue.isEmpty()) {
                AbstractTrainingNode<Regressor> node = queue.poll();
                if ((node.getImpurity() > 0.0) && (node.getDepth() < maxDepth) &&
                        (node.getWeightSum() >= minChildWeight)) {
                    if (numFeaturesInSplit != featureIDMap.size()) {
                        Util.randpermInPlace(originalIndices, localRNG);
                        System.arraycopy(originalIndices, 0, indices, 0, numFeaturesInSplit);
                    }
                    List<AbstractTrainingNode<Regressor>> nodes = node.buildTree(indices, localRNG,
                            getUseRandomSplitPoints());
                    // Use the queue as a stack to improve cache locality.
                    for (AbstractTrainingNode<Regressor> newNode : nodes) {
                        queue.addFirst(newNode);
                    }
                }
            }

            nodeMap.put(dimName,root.convertTree());
        }

        ModelProvenance provenance = new ModelProvenance(TreeModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        return new IndependentRegressionTreeModel("cart-tree", provenance, featureIDMap, outputIDInfo, false, nodeMap);
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("CARTRegressionTrainer(maxDepth=");
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