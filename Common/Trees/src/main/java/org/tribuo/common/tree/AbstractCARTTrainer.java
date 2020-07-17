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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.SkeletalTrainerProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.Collections;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Base class for {@link org.tribuo.Trainer}'s that use an approximation of the CART algorithm to build a decision tree.
 * <p>
 * See:
 * <pre>
 * J. Friedman, T. Hastie, &amp; R. Tibshirani.
 * "The Elements of Statistical Learning"
 * Springer 2001. <a href="http://web.stanford.edu/~hastie/ElemStatLearn/">PDF</a>
 * </pre>
 */
public abstract class AbstractCARTTrainer<T extends Output<T>> implements DecisionTreeTrainer<T> {

    /**
     * Default minimum weight of examples allowed in a leaf node.
     */
    public static final int MIN_EXAMPLES = 5;

    /**
     * Minimum weight of examples allowed in a leaf.
     */
    @Config(description="The minimum weight allowed in a child node.")
    protected float minChildWeight = MIN_EXAMPLES;

    /**
     * Maximum tree depth. Integer.MAX_VALUE indicates the depth is unlimited.
     */
    @Config(description="The maximum depth of the tree.")
    protected int maxDepth = Integer.MAX_VALUE;

    /**
     * Number of features to sample per split. 1 indicates all features are considered.
     */
    @Config(description="The fraction of features to consider in each split. 1.0f indicates all features are considered.")
    protected float fractionFeaturesInSplit = 1.0f;

    @Config(description="The RNG seed to use when sampling features in a split.")
    protected long seed = Trainer.DEFAULT_SEED;

    protected SplittableRandom rng;

    protected int trainInvocationCounter;

    /**
     * After calls to this superconstructor subclasses must call postConfig().
     * @param maxDepth The maximum depth of the tree.
     * @param minChildWeight The minimum child weight allowed.
     * @param fractionFeaturesInSplit The fraction of features to consider at each split.
     * @param seed The seed for the feature subsampling RNG.
     */
    protected AbstractCARTTrainer(int maxDepth, float minChildWeight, float fractionFeaturesInSplit, long seed) {
        this.maxDepth = maxDepth;
        this.fractionFeaturesInSplit = fractionFeaturesInSplit;
        this.minChildWeight = minChildWeight;
        this.seed = seed;
    }

    @Override
    public synchronized void postConfig() {
        this.rng = new SplittableRandom(seed);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public float getFractionFeaturesInSplit() {
        return fractionFeaturesInSplit;
    }

    @Override
    public TreeModel<T> train(Dataset<T> examples) {
        return train(examples, Collections.emptyMap());
    }

    @Override
    public TreeModel<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // Creates a new RNG, adds one to the invocation count.
        SplittableRandom localRNG;
        TrainerProvenance trainerProvenance;
        synchronized(this) {
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }

        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> outputIDInfo = examples.getOutputIDInfo();

        int numFeaturesInSplit = Math.min(Math.round(fractionFeaturesInSplit * featureIDMap.size()),featureIDMap.size());
        int[] indices;
        int[] originalIndices = new int[featureIDMap.size()];
        for (int i = 0; i < originalIndices.length; i++) {
            originalIndices[i] = i;
        }
        if (numFeaturesInSplit != featureIDMap.size()) {
            indices = new int[numFeaturesInSplit];
            // log
        } else {
            indices = originalIndices;
        }

        AbstractTrainingNode<T> root = mkTrainingNode(examples);
        Deque<AbstractTrainingNode<T>> queue = new LinkedList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            AbstractTrainingNode<T> node = queue.poll();
            if ((node.getDepth() < maxDepth) &&
                    (node.getNumExamples() > minChildWeight)) {
                if (numFeaturesInSplit != featureIDMap.size()) {
                    Util.randpermInPlace(originalIndices, localRNG);
                    System.arraycopy(originalIndices, 0, indices, 0, numFeaturesInSplit);
                }
                List<AbstractTrainingNode<T>> nodes = node.buildTree(indices);
                // Use the queue as a stack to improve cache locality.
                // Building depth first.
                for (AbstractTrainingNode<T> newNode : nodes) {
                    queue.addFirst(newNode);
                }
            }
        }

        ModelProvenance provenance = new ModelProvenance(TreeModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        return new TreeModel<>("cart-tree", provenance, featureIDMap, outputIDInfo, false, root.convertTree());
    }

    protected abstract AbstractTrainingNode<T> mkTrainingNode(Dataset<T> examples);

    protected static abstract class AbstractCARTTrainerProvenance extends SkeletalTrainerProvenance {
        private static final long serialVersionUID = 1L;

        protected <T extends Output<T>> AbstractCARTTrainerProvenance(AbstractCARTTrainer<T> host) {
            super(host);
        }

        protected AbstractCARTTrainerProvenance(Map<String,Provenance> map) {
            super(map);
        }
    }

}