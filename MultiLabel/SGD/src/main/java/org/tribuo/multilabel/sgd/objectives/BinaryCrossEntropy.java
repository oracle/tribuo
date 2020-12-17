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

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.util.SigmoidNormalizer;
import org.tribuo.multilabel.sgd.MultiLabelObjective;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.util.VectorNormalizer;

/**
 * A multilabel version of binary cross entropy loss which expects logits.
 * <p>
 * Generates a probabilistic model, and uses a {@link SigmoidNormalizer}.
 */
public final class BinaryCrossEntropy implements MultiLabelObjective {

    private static final VectorNormalizer normalizer = new SigmoidNormalizer();

    /**
     * Constructs a BinaryCrossEntropy objective.
     */
    public BinaryCrossEntropy() {}

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
        for (int i = 0; i < prediction.size(); i++) {
            double label = labels.get(i);
            double pred = densePred.get(i);
            double yhat = SigmoidNormalizer.sigmoid(pred);
            // numerically stable form of loss computation
            loss += Math.max(pred, 0) - (pred * label) + Math.log1p(Math.exp(-Math.abs(pred)));
            densePred.set(i,-(yhat - label));
        }
        return new Pair<>(loss,densePred);
    }

    @Override
    public VectorNormalizer getNormalizer() {
        return normalizer;
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
    public double threshold() {
        return 0.5;
    }

    @Override
    public String toString() {
        return "BinaryCrossEntropy";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"MultiLabelObjective");
    }
}
