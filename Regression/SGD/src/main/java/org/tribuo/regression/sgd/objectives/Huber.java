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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.regression.sgd.RegressionObjective;

import java.util.function.DoubleUnaryOperator;

/**
 * Huber loss, i.e., a mixture of l2 and l1 losses.
 */
public class Huber implements RegressionObjective {

    @Config(description="Cost beyond which the loss function is linear.")
    private double cost = 5;

    private DoubleUnaryOperator lossFunc;

    public Huber() {
        postConfig();
    }

    public Huber(double cost) {
        this.cost = cost;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        lossFunc = (a) -> {
            if (a > cost) {
                return (cost * a) - (0.5 * cost * cost);
            } else {
                return 0.5 * a * a;
            }
        };
    }

    @Override
    public Pair<Double, SGDVector> loss(DenseVector truth, SGDVector prediction) {
        DenseVector difference = truth.subtract(prediction);
        DenseVector absoluteDifference = difference.copy();
        absoluteDifference.foreachInPlace(Math::abs);

        double loss = absoluteDifference.reduce(0.0,lossFunc,Double::sum);
        difference.foreachInPlace((a) -> {if (Math.abs(a) > cost) { return Double.compare(a,0.0)*cost; } else { return a; }});
        return new Pair<>(loss,difference);
    }

    @Override
    public String toString() {
        return "Huber(cost="+cost+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"RegressionObjective");
    }
}
