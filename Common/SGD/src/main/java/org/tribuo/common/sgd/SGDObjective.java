/*
 * Copyright (c) 2020, 2023, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.sgd;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.SGDVector;

/**
 * An interface for a loss function that can produce the loss and gradient incurred by
 * a single prediction.
 * @param <T> The type of a single ground truth output at training time.
 * @param <U> The type of a batch of ground truth outputs at training time.
 */
public interface SGDObjective<T,U> extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Scores a prediction, returning the loss and a vector of per output dimension gradients.
     *
     * @param truth      The true output.
     * @param prediction The prediction for each dimension.
     * @return The score and per dimension gradient.
     */
    Parameters.LossAndGrad lossAndGradient(T truth, SGDVector prediction);

    /**
     * Scores a batch of predictions, returning the loss and a matrix of per output dimension gradients.
     *
     * @param truth      The true outputs.
     * @param prediction The prediction matrix.
     * @return The score and per dimension gradient.
     */
    Parameters.BatchLossAndGrad batchLossAndGradient(U truth, DenseMatrix prediction);

    double loss(T truth, SGDVector prediction);

    double[] batchLoss(U truth, DenseMatrix prediction);
}
