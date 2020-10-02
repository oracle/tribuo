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
import org.tribuo.SparseTrainer;
import org.tribuo.Trainer;
import org.tribuo.WeightedExamples;

/**
 * A tag interface for a {@link Trainer} so the random forests trainer can check if it's actually a tree.
 */
public interface DecisionTreeTrainer<T extends Output<T>> extends SparseTrainer<T>, WeightedExamples {

    /**
     * Returns the feature subsampling rate.
     * @return The feature subsampling rate.
     */
    public float getFractionFeaturesInSplit();

    /**
     * Returns whether to choose split points for features at random.
     * @return Whether to choose split points for features at random.
     */
    public boolean getUseRandomSplitPoints();

    /**
     * Returns the minimum decrease in impurity necessary to split a node.
     * @return The minimum decrease in impurity necessary to split a node.
     */
    public float getMinImpurityDecrease();
}
