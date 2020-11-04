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

package org.tribuo.classification.dtree.impurity;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

/**
 * A log_e entropy impurity measure.
 */
public class Entropy implements LabelImpurity {

    @Override
    public double impurityNormed(double[] input) {
        double score = 0.0;

        for (int i = 0; i < input.length; i++) {
            double d = input[i];
            if (Math.abs(d) > 1e-10) {
                score -= d * Math.log(d);
            }
        }

        return score;
    }

    @Override
    public String toString() {
        return "Entropy";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"LabelImpurity");
    }
}
