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

package org.tribuo.multilabel.evaluation;

import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.ConfusionMatrix;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;

import java.util.List;
import java.util.Set;
import java.util.function.Function;

/**
 * A {@link ConfusionMatrix} which accepts {@link MultiLabel}s.
 *
 * <p>
 * In a multi-label confusion matrix M,
 * <pre>
 * tn = M[:, 0, 0]
 * fn = M[:, 1, 0]
 * tp = M[:, 1, 1]
 * fp = M[:, 0, 1]
 * </pre>
 * <p>
 * For class-wise values,
 * <pre>
 * tn(class i) = M[i, 0, 0]
 * fn(class i) = M[i, 1, 0]
 * tp(class i) = M[i, 1, 1]
 * fp(class i) = M[i, 0, 1]
 * </pre>
 */
public final class MultiLabelConfusionMatrix implements ConfusionMatrix<MultiLabel> {

    private final ImmutableOutputInfo<MultiLabel> domain;
    private final DenseMatrix[] mcm;
    private final DenseMatrix confusion;

    public MultiLabelConfusionMatrix(Model<MultiLabel> model, List<Prediction<MultiLabel>> predictions) {
        this(model.getOutputIDInfo(), predictions);
    }

    MultiLabelConfusionMatrix(ImmutableOutputInfo<MultiLabel> domain, List<Prediction<MultiLabel>> predictions) {
        this.domain = domain;
        ConfusionMatrixTuple tab = tabulate(domain, predictions);
        this.mcm = tab.mcm;
        this.confusion = tab.confusion;
    }

    @Override
    public double support(MultiLabel cls) {
        double total = 0d;
        for (Label label : cls.getLabelSet()) {
            int ix = getDomain().getID(new MultiLabel(label));
            /*
            mcm[i] =
            [tn, fn]
            [fp, tp]

            support = false negatives + true positives

            false neg => ground truth was [label] but we predicted something else
            true pos  => ground truth was [label] and we predicted [label]

            (whereas: false pos => ground truth was NOT [label] but we predicted [label])

            so

            support = false neg + true pos = mcm[i, 0, 1] + mcm[i, 1, 1] = mcm[i, :, 1].sum()
             */
            total += mcm[ix].getColumn(1).sum();
        }
        return total;
    }

    @Override
    public ImmutableOutputInfo<MultiLabel> getDomain() {
        return domain;
    }

    @Override
    public double support() {
        double total = 0d;
        for (int i = 0; i < domain.size(); i++) {
            total += mcm[i].getColumn(1).sum();
        }
        return total;
    }

    @Override
    public double tp(MultiLabel cls) {
        return compute(cls, (cm) -> cm.get(1, 1));
    }

    @Override
    public double fp(MultiLabel cls) {
        return compute(cls, (cm) -> cm.get(0, 1));
    }

    @Override
    public double fn(MultiLabel cls) {
        return compute(cls, (cm) -> cm.get(1, 0));
    }

    @Override
    public double tn(MultiLabel cls) {
        return compute(cls, (cm) -> cm.get(0, 0));
    }

    private double compute(MultiLabel cls, Function<DenseMatrix, Double> getter) {
        double total = 0d;
        for (Label label : cls.getLabelSet()) {
            int i = domain.getID(new MultiLabel(label.getLabel()));
            //
            // When input class is not in the domain, ID will be -1.
            if (i < 0) {
                continue;
            }
            DenseMatrix cm = mcm[i];
            total += getter.apply(cm);
        }
        return total;
    }

    @Override
    public double confusion(MultiLabel predicted, MultiLabel truth) {
        double total = 0d;
        Set<Label> trueSet = truth.getLabelSet();
        Set<Label> predSet = predicted.getLabelSet();
        for (Label predLabel : predSet) {
            int idx = domain.getID(new MultiLabel(predLabel.getLabel()));
            for (Label trueLabel : trueSet) {
                int jdx = domain.getID(new MultiLabel(trueLabel.getLabel()));
                total += this.confusion.get(idx, jdx);
            }
        }
        return total;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < mcm.length; i++) {
            DenseMatrix cm = mcm[i];
            sb.append(cm.toString());
            sb.append("\n");
        }
        sb.append("]");
        return sb.toString();
    }

    static ConfusionMatrixTuple tabulate(ImmutableOutputInfo<MultiLabel> domain, List<Prediction<MultiLabel>> predictions) {
        // this just keeps track of how many times [class x] was predicted to be [class y]
        DenseMatrix confusion = new DenseMatrix(domain.size(), domain.size());

        DenseMatrix[] mcm = new DenseMatrix[domain.size()];
        for (int i = 0; i < domain.size(); i++) {
            mcm[i] = new DenseMatrix(2, 2);
        }

        int predIndex = 0;
        for (Prediction<MultiLabel> prediction : predictions) {
            MultiLabel predictedOutput = prediction.getOutput();
            MultiLabel trueOutput = prediction.getExample().getOutput();
            if (trueOutput.equals(MultiLabelFactory.UNKNOWN_MULTILABEL)) {
                throw new IllegalArgumentException("The sentinel Unknown MultiLabel was used as a ground truth label at prediction number " + predIndex);
            } else if (predictedOutput.equals(MultiLabelFactory.UNKNOWN_MULTILABEL)) {
                throw new IllegalArgumentException("The sentinel Unknown MultiLabel was predicted by the model at prediction number " + predIndex);
            }

            Set<Label> trueSet = trueOutput.getLabelSet();
            Set<Label> predSet = predictedOutput.getLabelSet();

            //
            // Count true positives and false positives
            for (Label pred : predSet) {
                int idx = domain.getID(new MultiLabel(pred.getLabel()));
                if (trueSet.contains(pred)) {
                    //
                    // true positive: mcm[i, 1, 1]++
                    mcm[idx].add(1, 1, 1d);
                } else {
                    //
                    // false positive: mcm[i, 1, 0]++
                    mcm[idx].add(1, 0, 1d);
                }
            }

            //
            // Count false negatives and populate the confusion table
            for (Label trueLabel : trueSet) {
                int idx = domain.getID(new MultiLabel(trueLabel.getLabel()));
                if (idx < 0) {
                    throw new IllegalArgumentException("Unknown label '" + trueLabel.getLabel() + "' found in the ground truth labels at prediction number " + predIndex
                            + ", this label is not known by the model which made the predictions.");
                }

                //
                // Doing two things in this loop:
                // 1) Checking if predSet contains trueLabel
                // 2) Counting the # of times [trueLabel] was predicted to be [predLabel] to populate the confusion table
                boolean found = false;
                for (Label predLabel : predSet) {
                    int jdx = domain.getID(new MultiLabel(predLabel.getLabel()));
                    confusion.add(jdx, idx, 1d);

                    if (predLabel.equals(trueLabel)) {
                        found = true;
                    }
                }

                if (!found) {
                    //
                    // false negative: mcm[i, 0, 1]++
                    mcm[idx].add(0, 1, 1d);
                }
                // else { true positive: already counted }
            }

            //
            // True negatives everywhere else
            for (MultiLabel multilabel : domain.getDomain()) {
                Set<Label> labels = multilabel.getLabelSet();
                for (Label label : labels) {
                    if (!trueSet.contains(label) && !predSet.contains(label)) {
                        int ix = domain.getID(new MultiLabel(label));
                        mcm[ix].add(0, 0, 1d);
                    }
                }
            }
            predIndex++;
        }

        return new ConfusionMatrixTuple(mcm, confusion);
    }

    /**
     * It's a record, ooops not yet, we don't require Java 14.
     */
    static final class ConfusionMatrixTuple {
        final DenseMatrix[] mcm;
        final DenseMatrix confusion;
        ConfusionMatrixTuple(DenseMatrix[] mcm, DenseMatrix confusion) {
            this.mcm = mcm;
            this.confusion = confusion;
        }

        DenseMatrix[] getMCM() {
            return mcm;
        }
    }
}