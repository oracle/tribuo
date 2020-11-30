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

package org.tribuo.classification.sgd;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.common.sgd.SGDObjective;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.util.VectorNormalizer;

/**
 * An interface for single label prediction objectives.
 * <p>
 * An objective knows if it generates a probabilistic model or not,
 * and what kind of normalization needs to be applied to produce probability values.
 */
public interface LabelObjective extends SGDObjective<Integer> {

    /**
     * Scores a prediction, returning the loss and a vector of per label gradients.
     *
     * @deprecated In 4.1, to migrate to the new name {@link #lossAndGradient}.
     * @param truth      The true label id.
     * @param prediction The prediction for each label id.
     * @return The score and per label gradient.
     */
    @Deprecated
    Pair<Double, SGDVector> valueAndGradient(int truth, SGDVector prediction);

    @Override
    default Pair<Double, SGDVector> lossAndGradient(Integer truth, SGDVector prediction) {
        return valueAndGradient(truth, prediction);
    }

    /**
     * Generates a new {@link VectorNormalizer} which normalizes the predictions into [0,1].
     * @return The vector normalizer for this objective.
     */
    public VectorNormalizer getNormalizer();

    /**
     * Does the objective function score probabilities or not?
     * @return boolean.
     */
    public boolean isProbabilistic();

}
