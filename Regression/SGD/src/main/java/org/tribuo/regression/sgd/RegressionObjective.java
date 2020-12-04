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

package org.tribuo.regression.sgd;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.common.sgd.SGDObjective;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;

/**
 * An interface for regression objectives.
 */
public interface RegressionObjective extends SGDObjective<DenseVector> {

    /**
     * Scores a prediction, returning the loss.
     * @deprecated In 4.1 to move to the new name, lossAndGradient.
     * @param truth The true regression value.
     * @param prediction The predicted regression value.
     * @return A pair with the loss and gradient.
     */
    @Deprecated
    public Pair<Double, SGDVector> loss(DenseVector truth, SGDVector prediction);

    @Override
    default public Pair<Double, SGDVector> lossAndGradient(DenseVector truth, SGDVector prediction) {
        return loss(truth, prediction);
    }

}
