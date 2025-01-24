/*
 * Copyright (c) 2015, 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.sgd.objectives;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.classification.sgd.LabelObjective;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.util.ExpNormalizer;
import org.tribuo.math.util.VectorNormalizer;

/**
 * A multiclass version of the log loss.
 * <p>
 * Generates a probabilistic model, and uses an {@link ExpNormalizer}.
 */
public class LogMulticlass implements LabelObjective {

    private final VectorNormalizer normalizer = new ExpNormalizer();

    /**
     * Constructs a multiclass log loss.
     */
    public LogMulticlass() {}

    /**
     * Returns a {@link org.tribuo.math.Parameters.LossAndGrad} containing the loss and per label gradients.
     * <p>
     * The prediction vector is transformed to produce the per label gradient and returned.
     * @param truth The true label id
     * @param prediction The prediction for each label id
     * @return The score and per label gradient.
     */
    @Override
    public Parameters.LossAndGrad lossAndGradient(Integer truth, SGDVector prediction) {
        prediction.normalize(normalizer);
        double loss = -Math.log(prediction.get(truth));
        prediction.scaleInPlace(-1.0);
        prediction.add(truth,1.0);
        return new Parameters.LossAndGrad(loss,prediction);
    }

    @Override
    public Parameters.BatchLossAndGrad batchLossAndGradient(int[] truth, DenseMatrix prediction) {
        prediction.normalizeRows(normalizer);
        prediction.scaleInPlace(-1.0);
        double[] loss = new double[truth.length];
        for (int i = 0; i < truth.length; i++) {
            loss[i] = -Math.log(prediction.get(i, truth[i]) * -1.0);
            prediction.add(i, truth[i], 1.0);
        }
        return new Parameters.BatchLossAndGrad(loss,prediction);
    }

    @Override
    public double loss(Integer truth, SGDVector prediction) {
        prediction.normalize(normalizer);
        return -Math.log(prediction.get(truth));
    }

    @Override
    public double[] batchLoss(int[] truth, DenseMatrix prediction) {
        prediction.normalizeRows(normalizer);
        double[] loss = new double[truth.length];
        for (int i = 0; i < truth.length; i++) {
            loss[i] = -Math.log(prediction.get(i, truth[i]));
        }
        return loss;
    }

    @Override
    public VectorNormalizer getNormalizer() {
        return new ExpNormalizer();
    }

    /**
     * Returns true.
     * @return True.
     */
    @Override
    public boolean isProbabilistic() {
        return true;
    }

    @Override
    public String toString() {
        return "LogMulticlass";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"LabelObjective");
    }
}
