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

/**
 * Base class for decision tree nodes used at training time.
 */
public abstract class AbstractTrainingNode<T extends Output<T>> implements Node<T> {

    /**
     * Default buffer size used in the split operation.
     */
    protected static final int DEFAULT_SIZE = 16;

    protected final int depth;

    protected final int numExamples;

    protected boolean split;

    protected int splitID;

    protected double splitValue;
    
    protected AbstractTrainingNode<T> greaterThan;
    
    protected AbstractTrainingNode<T> lessThanOrEqual;

    /**
     * Builds an abstract training node.
     * @param depth The depth of this node.
     * @param numExamples The number of examples in this node.
     */
    protected AbstractTrainingNode(int depth, int numExamples) {
        this.depth = depth;
        this.numExamples = numExamples;
    }

    public abstract List<AbstractTrainingNode<T>> buildTree(int[] indices);

    /**
     * Converts a tree from a training representation to the final inference time representation.
     * @return The converted subtree.
     */
    public abstract Node<T> convertTree();

    /**
     * The depth of this node in the tree.
     * @return The depth.
     */
    public int getDepth() {
        return depth;
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

    /**
     * The number of training examples in this node.
     * @return The number of training examples in this node.
     */
    public int getNumExamples() {
        return numExamples;
    }

    @Override
    public boolean isLeaf() {
        return !split;
    }

    @Override
    public Node<T> copy() {
        throw new UnsupportedOperationException("Copy is not supported on training nodes.");
    }
}
