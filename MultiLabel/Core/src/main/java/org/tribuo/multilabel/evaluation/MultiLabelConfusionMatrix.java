/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A {@link ConfusionMatrix} which accepts {@link MultiLabel}s.
 *
 * <p>
 * In a multi-label confusion matrix M,
 * <pre>
 * tn = M[:, 0, 0]
 * fn = M[:, 0, 1]
 * fp = M[:, 1, 0]
 * tp = M[:, 1, 1]
 * </pre>
 * <p>
 * For class-wise values,
 * <pre>
 * tn(class i) = M[i, 0, 0]
 * fn(class i) = M[i, 0, 1]
 * fp(class i) = M[i, 1, 0]
 * tp(class i) = M[i, 1, 1]
 * </pre>
 */
public final class MultiLabelConfusionMatrix implements ConfusionMatrix<MultiLabel> {

    private final ImmutableOutputInfo<MultiLabel> domain;
    private final DenseMatrix[] mcm;
    private final DenseMatrix confusion;
    private final Set<MultiLabel> observed;

    private List<MultiLabel> labelOrder;


    /**
     * Constructs a multi-label confusion matrix for the specified model and predictions.
     * @param model The model.
     * @param predictions The predictions.
     */
    public MultiLabelConfusionMatrix(Model<MultiLabel> model, List<Prediction<MultiLabel>> predictions) {
        this(model.getOutputIDInfo(), predictions);
    }

    MultiLabelConfusionMatrix(ImmutableOutputInfo<MultiLabel> domain, List<Prediction<MultiLabel>> predictions) {
        this.domain = domain;
        this.labelOrder = Collections.unmodifiableList(new ArrayList<>(domain.getDomain()));
        ConfusionMatrixTuple tab = tabulate(domain, predictions);
        this.mcm = tab.mcm;
        this.confusion = tab.confusion;
        this.observed = tab.observed;
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
    public Set<MultiLabel> observed() {
        return Collections.unmodifiableSet(observed);
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
        return compute(cls, (cm) -> cm.get(1, 0));
    }

    @Override
    public double fn(MultiLabel cls) {
        return compute(cls, (cm) -> cm.get(0, 1));
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

    /**
     * Sets the label order used in {@link #toString}.
     * <p>
     * If the label order is a subset of the labels in the domain, only the
     * labels present in the label order will be displayed.
     *
     * @param labelOrder The label order to use.
     */
    @Override
    public void setLabelOrder(List<MultiLabel> labelOrder) {
        if (labelOrder == null || labelOrder.isEmpty()) {
            throw new IllegalArgumentException("Label order must be non-null and non-empty.");
        }
        this.labelOrder = Collections.unmodifiableList(new ArrayList<>(labelOrder));
    }

    /**
     * Gets the current label order.
     * <p>
     * If the label order is a subset of the labels in the domain, only the
     * labels present in the label order will be displayed.
     *
     * @return The label order.
     */
    public List<MultiLabel> getLabelOrder() {
        return labelOrder;
    }

    @Override
    public String toString() {
        return labelOrder.stream()
            .map(multiLabel -> {
                  final int tp = (int) tp(multiLabel);
                  final int fn = (int) fn(multiLabel);
                  final int fp = (int) fp(multiLabel);
                  final int tn = (int) tn(multiLabel);
                  return String.join("\n",
                      multiLabel.toString(),
                      String.format("    [tn: %,d fn: %,d]", tn, fn),
                      String.format("    [fp: %,d tp: %,d]", fp, tp));
                }
            ).collect(Collectors.joining("\n"));
    }

    static ConfusionMatrixTuple tabulate(ImmutableOutputInfo<MultiLabel> domain, List<Prediction<MultiLabel>> predictions) {
        // this just keeps track of how many times [class x] was predicted to be [class y]
        DenseMatrix confusion = new DenseMatrix(domain.size(), domain.size());

        Set<MultiLabel> observed = new HashSet<>();

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
                observed.add(new MultiLabel(pred));
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
                observed.add(new MultiLabel(trueLabel));
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

        return new ConfusionMatrixTuple(mcm, confusion, observed);
    }

    /**
     * It's a record, ooops not yet, we don't require Java 14.
     */
    static final class ConfusionMatrixTuple {
        final DenseMatrix[] mcm;
        final DenseMatrix confusion;
        final Set<MultiLabel> observed;
        ConfusionMatrixTuple(DenseMatrix[] mcm, DenseMatrix confusion, Set<MultiLabel> observed) {
            this.mcm = mcm;
            this.confusion = confusion;
            this.observed = observed;
        }

        DenseMatrix[] getMCM() {
            return mcm;
        }
    }
}