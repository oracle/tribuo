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

import org.tribuo.Output;
import org.tribuo.math.la.SparseVector;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Base class for decision tree nodes used at training time.
 */
public abstract class AbstractTrainingNode<T extends Output<T>> implements Node<T> {

    protected static final int DEFAULT_SIZE = 16;

    protected final int depth;

    protected final int numExamples;

    protected final LeafDeterminer leafDeterminer;

    protected boolean split;

    protected int splitID;

    protected double splitValue;

    protected double impurityScore;
    
    protected Node<T> greaterThan;
    
    protected Node<T> lessThanOrEqual;

    protected AbstractTrainingNode(int depth, int numExamples, LeafDeterminer leafDeterminer) {
        this.depth = depth;
        this.numExamples = numExamples;
        this.leafDeterminer = leafDeterminer;
    }

    /**
     * Builds next level of a tree.
     * @param featureIDs Indices of the features available in this split.
     * @param rng Splittable random number generator.
     * @param useRandomSplitPoints Whether to choose split points for features at random.
     * @return A possibly empty list of TrainingNodes.
     */
    public abstract List<AbstractTrainingNode<T>> buildTree(int[] featureIDs, SplittableRandom rng,
                                                            boolean useRandomSplitPoints);

    public abstract Node<T> convertTree();

    public abstract float getWeightSum();

    public int getDepth() {
        return depth;
    }

    public int getNumExamples() {
        return numExamples;
    }

    public boolean shouldMakeLeaf(double impurityScore, float weightSum) {
        return ((impurityScore == 0.0) ||
                (depth + 1 >= leafDeterminer.getMaxDepth()) ||
                (weightSum < leafDeterminer.getMinChildWeight()));
    }

    public SplitNode<T> mkSplitNode() {
        Node<T> newGreaterThan = greaterThan;
        Node<T> newLessThan = lessThanOrEqual;

        // split node
        if (greaterThan instanceof AbstractTrainingNode) {
            AbstractTrainingNode<T> abstractGreaterThan = (AbstractTrainingNode<T>) greaterThan;
            newGreaterThan = abstractGreaterThan.convertTree();
        }

        if (lessThanOrEqual instanceof AbstractTrainingNode) {
            AbstractTrainingNode<T> abstractLessThan = (AbstractTrainingNode<T>) lessThanOrEqual;
            newLessThan = abstractLessThan.convertTree();
        }
        return new SplitNode<>(splitValue,splitID,getImpurity(),newGreaterThan,newLessThan);
    }

    @Override
    public Node<T> getNextNode(SparseVector example) {
        if (split) {
            double feature = example.get(splitID);
            if (feature > splitValue) {
                return greaterThan;
            } else {
                return lessThanOrEqual;
            }
        } else {
            return null;
        }
    }

    @Override
    public boolean isLeaf() {
        return !split;
    }

    @Override
    public Node<T> copy() {
        throw new UnsupportedOperationException("Copy is not supported on training nodes.");
    }

    /**
     * Contains parameters needed to determine whether a node is a leaf.
     */
    public static class LeafDeterminer {
        private final int maxDepth;
        private final float minChildWeight;
        private final float scaledMinImpurityDecrease;

        public LeafDeterminer(int maxDepth, float minChildWeight, float scaledMinImpurityDecrease) {
            this.maxDepth = maxDepth;
            this.minChildWeight = minChildWeight;
            this.scaledMinImpurityDecrease = scaledMinImpurityDecrease;
        }

        public int getMaxDepth() {
            return maxDepth;
        }

        public float getMinChildWeight() {
            return minChildWeight;
        }

        public float getScaledMinImpurityDecrease() {
            return scaledMinImpurityDecrease;
        }
    }

}