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

package org.tribuo.classification.sgd.objectives;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.classification.sgd.LabelObjective;
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

    @Deprecated
    @Override
    public Pair<Double, SGDVector> valueAndGradient(int truth, SGDVector prediction) {
        return lossAndGradient(truth, prediction);
    }

    /**
     * Returns a {@link Pair} of {@link Double} and {@link SGDVector} representing the loss
     * and per label gradients respectively.
     * <p>
     * The prediction vector is transformed to produce the per label gradient and returned.
     * @param truth The true label id
     * @param prediction The prediction for each label id
     * @return A Pair of the score and per label gradient.
     */
    @Override
    public Pair<Double,SGDVector> lossAndGradient(Integer truth, SGDVector prediction) {
        prediction.normalize(normalizer);
        double loss = Math.log(prediction.get(truth));
        prediction.scaleInPlace(-1.0);
        prediction.add(truth,1.0);
        return new Pair<>(loss,prediction);
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
