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

package org.tribuo.multilabel.sgd.objectives;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.la.DenseVector;
import org.tribuo.multilabel.sgd.MultiLabelObjective;
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
public final class Hinge implements MultiLabelObjective {

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

    /**
     * Returns a {@link Pair} of {@link Double} and {@link SGDVector} representing the loss
     * and per label gradients respectively.
     * @param truth The true label id.
     * @param prediction The prediction for each label id.
     * @return The loss and per label gradient.
     */
    @Override
    public Pair<Double,SGDVector> lossAndGradient(SGDVector truth, SGDVector prediction) {
        DenseVector labels, densePred;
        if (truth instanceof SparseVector) {
            labels = ((SparseVector) truth).densify();
        } else {
            labels = (DenseVector) truth;
        }
        if (prediction instanceof SparseVector) {
            densePred = ((SparseVector) prediction).densify();
        } else {
            densePred = (DenseVector) prediction;
        }
        double loss = 0.0;
        for (int i = 0; i < labels.size(); i++) {
           double lbl = labels.get(i) == 0.0 ? -1 : 1.0;
           double pred = densePred.get(i);
           double score = lbl * pred;
           if (score < margin) {
               densePred.set(i, lbl);
           } else {
               densePred.set(i, 0.0);
           }
           loss += Math.max(0.0,margin - score);
        }
        return new Pair<>(loss,densePred);
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
    public double threshold() {
        return 0;
    }

    @Override
    public String toString() {
        return "MultiLabelHinge(margin="+margin+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"MultiLabelObjective");
    }
}
