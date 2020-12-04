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

package org.tribuo.regression.sgd.objectives;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.regression.sgd.RegressionObjective;

/**
 * Squared loss, i.e., l2.
 */
public class SquaredLoss implements RegressionObjective {

    /**
     * Constructs a SquaredLoss.
     */
    public SquaredLoss() {}

    @Deprecated
    @Override
    public Pair<Double, SGDVector> loss(DenseVector truth, SGDVector prediction) {
        return lossAndGradient(truth, prediction);
    }

    @Override
    public Pair<Double, SGDVector> lossAndGradient(DenseVector truth, SGDVector prediction) {
        DenseVector difference = truth.subtract(prediction);
        double loss = difference.reduce(0.0,(a) -> 0.5*a*a,Double::sum);
        return new Pair<>(loss,difference);
    }

    @Override
    public String toString() {
        return "SquaredLoss";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"RegressionObjective");
    }
}
