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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.classification.sgd.LabelObjective;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.util.NoopNormalizer;
import org.tribuo.math.util.VectorNormalizer;

/**
 * Hinge loss, scores the correct value margin and any incorrect predictions -margin.
 * By default the margin is 1.0.
 * <p>
 * The Hinge loss does not generate a probabilistic model, and uses a {@link NoopNormalizer}.
 */
public class Hinge implements LabelObjective {

    @Config(description="The classification margin.")
    private double margin = 1.0;

    /**
     * Construct a hinge objective with the supplied margin.
     * @param margin The margin to use.
     */
    public Hinge(double margin) {
        this.margin = margin;
    }

    /**
     * Construct a hinge objective with a margin of 1.0.
     */
    public Hinge() {
        this(1.0);
    }

    @Deprecated
    @Override
    public Pair<Double, SGDVector> valueAndGradient(int truth, SGDVector prediction) {
        return lossAndGradient(truth, prediction);
    }

    /**
     * Returns a {@link Pair} of {@link Double} and {@link SGDVector} representing the loss
     * and per label gradients respectively.
     * @param truth The true label id.
     * @param prediction The prediction for each label id.
     * @return The loss and per label gradient.
     */
    @Override
    public Pair<Double,SGDVector> lossAndGradient(Integer truth, SGDVector prediction) {
        prediction.add(truth,-margin);
        int predIndex = prediction.indexOfMax();

        if (truth == predIndex) {
            return new Pair<>(0.0, SparseVector.createSparseVector(prediction.size(),new int[0], new double[0]));
        } else {
            int[] indices = new int[2];
            double[] values = new double[2];
            if (truth < predIndex) {
                indices[0] = truth;
                values[0] = margin;
                indices[1] = predIndex;
                values[1] = -margin;
            } else {
                indices[0] = predIndex;
                values[0] = -margin;
                indices[1] = truth;
                values[1] = margin;
            }
            SparseVector output = SparseVector.createSparseVector(prediction.size(),indices,values);
            double loss = prediction.get(truth) - prediction.get(predIndex);
            return new Pair<>(loss,output);
        }
    }

    /**
     * Returns a new {@link NoopNormalizer}.
     * @return The vector normalizer.
     */
    @Override
    public VectorNormalizer getNormalizer() {
        return new NoopNormalizer();
    }

    /**
     * Returns false.
     * @return False.
     */
    @Override
    public boolean isProbabilistic() {
        return false;
    }

    @Override
    public String toString() {
        return "Hinge(margin="+margin+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"LabelObjective");
    }
}
