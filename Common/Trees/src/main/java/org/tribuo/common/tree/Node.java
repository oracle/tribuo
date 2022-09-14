/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

import java.io.Serializable;

/**
 * A node in a decision tree.
 */
public interface Node<T extends Output<T>> extends Serializable {

    /**
     * Is it a leaf node?
     * @return True if it's a leaf node.
     */
    public boolean isLeaf();

    /**
     * Returns the next node in the tree based on the supplied example, or null if it's a leaf.
     * @param example The example.
     * @return The next node down in the tree.
     */
    public Node<T> getNextNode(SparseVector example);

    /**
     * The impurity score of this node.
     * @return The node impurity.
     */
    public double getImpurity();

    /**
     * Copies the node and it's children.
     * @return A deep copy.
     */
    public Node<T> copy();
}
