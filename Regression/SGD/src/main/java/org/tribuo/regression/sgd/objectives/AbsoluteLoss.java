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

package org.tribuo.regression.sgd.objectives;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.regression.sgd.RegressionObjective;

/**
 * Absolute loss (i.e., l1).
 */
public class AbsoluteLoss implements RegressionObjective {

    /**
     * Constructs an absolute loss.
     */
    public AbsoluteLoss() {}

    @Override
    public Parameters.LossAndGrad lossAndGradient(DenseVector truth, SGDVector prediction) {
        DenseVector difference = truth.subtract(prediction);
        double startValue = - (0.5 * difference.size());
        double loss = difference.reduce(startValue, Math::abs, Double::sum);
        difference.foreachInPlace((a) -> Double.compare(a,0.0));
        return new Parameters.LossAndGrad(loss, difference);
    }

    @Override
    public Parameters.BatchLossAndGrad batchLossAndGradient(DenseMatrix truth, DenseMatrix prediction) {
        DenseMatrix difference = truth.subtract(prediction);
        double startValue = - (0.5 * difference.getDimension2Size());
        DenseVector loss = difference.reduceRows(startValue, Math::abs, Double::sum);
        difference.foreachInPlace((a) -> Double.compare(a,0.0));
        return new Parameters.BatchLossAndGrad(loss.toArray(), difference);
    }

    @Override
    public double loss(DenseVector truth, SGDVector prediction) {
        DenseVector difference = truth.subtract(prediction);
        double startValue = - (0.5 * difference.size());
        return difference.reduce(startValue, Math::abs, Double::sum);
    }

    @Override
    public double[] batchLoss(DenseMatrix truth, DenseMatrix prediction) {
        DenseMatrix difference = truth.subtract(prediction);
        double startValue = - (0.5 * difference.getDimension2Size());
        DenseVector loss = difference.reduceRows(startValue, Math::abs, Double::sum);
        return loss.toArray();
    }

    @Override
    public String toString() {
        return "AbsoluteLoss";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"RegressionObjective");
    }
}
