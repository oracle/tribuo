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

package org.tribuo.classification.evaluation;

import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.math.la.DenseMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.ToDoubleFunction;
import java.util.logging.Logger;

/**
 * A confusion matrix for {@link Label}s.
 * <p>
 * We interpret it as follows:
 *
 * {@code
 * C[i, j] = k
 * }
 *
 * means "the TRUE class 'j' was PREDICTED to be class 'i' a total of 'k' times".
 *
 * <p>
 * In other words, the row indices correspond to the model's predictions, and the column indices correspond to
 * the ground truth.
 * </p>
 */
public final class LabelConfusionMatrix implements ConfusionMatrix<Label> {

    private static final Logger logger = Logger.getLogger(LabelConfusionMatrix.class.getName());

    private final ImmutableOutputInfo<Label> domain;

    private final int total;
    private final Map<Label, Double> occurrences;

    private final Set<Label> observed;

    private final DenseMatrix cm;

    private List<Label> labelOrder;

    /**
     * Creates a confusion matrix from the supplied predictions, using the label info
     * from the supplied model.
     *
     * @param model       The model to use for the label information.
     * @param predictions The predictions.
     */
    public LabelConfusionMatrix(Model<Label> model, List<Prediction<Label>> predictions) {
        this(model.getOutputIDInfo(), predictions);
    }

    /**
     * Creates a confusion matrix from the supplied predictions and label info.
     *
     * @param domain      The label information.
     * @param predictions The predictions.
     * @throws IllegalArgumentException If the domain doesn't contain all the predictions.
     */
    public LabelConfusionMatrix(ImmutableOutputInfo<Label> domain, List<Prediction<Label>> predictions) {
        this.domain = domain;
        this.total = predictions.size();
        this.cm = new DenseMatrix(domain.size(), domain.size());
        this.occurrences = new HashMap<>();
        this.observed = new HashSet<>();
        this.labelOrder = Collections.unmodifiableList(new ArrayList<>(domain.getDomain()));
        tabulate(predictions);
    }

    /**
     * Aggregate the predictions into this confusion matrix.
     *
     * @param predictions The predictions to aggregate.
     */
    private void tabulate(List<Prediction<Label>> predictions) {
        predictions.forEach(prediction -> {
            Label y = prediction.getExample().getOutput();
            Label p = prediction.getOutput();
            //
            // Check that the ground truth label is valid
            if (y.getLabel().equals(Label.UNKNOWN)) {
                throw new IllegalArgumentException("Prediction with unknown ground truth. Unable to evaluate.");
            }
            occurrences.merge(y, 1d, Double::sum);
            observed.add(y);
            observed.add(p);
            int iy = getIDOrThrow(y);
            int ip = getIDOrThrow(p);
            cm.add(ip, iy, 1d);
        });
    }

    @Override
    public ImmutableOutputInfo<Label> getDomain() {
        return domain;
    }

    @Override
    public Set<Label> observed() {
        return Collections.unmodifiableSet(observed);
    }

    @Override
    public double support() {
        return total;
    }

    @Override
    public double support(Label label) {
        return occurrences.getOrDefault(label, 0d);
    }

    @Override
    public double tp(Label cls) {
        return compute(cls, (i) -> cm.get(i, i));
    }

    @Override
    public double fp(Label cls) {
        // Row-wise sum less true positives
        return compute(cls, i -> cm.rowSum(i) - cm.get(i, i));
    }

    @Override
    public double fn(Label cls) {
        // Column-wise sum less true positives
        return compute(cls, i -> cm.columnSum(i) - cm.get(i, i));
    }

    @Override
    public double tn(Label cls) {
        int n = getDomain().size();
        int i = getDomain().getID(cls);
        double total = 0d;
        for (int j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            for (int k = 0; k < n; k++) {
                if (k == i) {
                    continue;
                }
                total += cm.get(j, k);
            }
        }
        return total;
    }

    @Override
    public double confusion(Label predicted, Label trueClass) {
        int i = getDomain().getID(predicted);
        int j = getDomain().getID(trueClass);
        return cm.get(i, j);
    }

    /**
     * A convenience method for extracting the appropriate label statistic.
     *
     * @param cls    The label to check.
     * @param getter The get function which accepts a label id.
     * @return The statistic for that label id.
     */
    private double compute(Label cls, ToDoubleFunction<Integer> getter) {
        int i = getDomain().getID(cls);
        if (i < 0) {
            logger.fine("Unknown Label " + cls);
            return 0d;
        }
        return getter.applyAsDouble(i);
    }

    /**
     * Gets the id for the supplied label, or throws an {@link IllegalArgumentException} if it's
     * an unknown label.
     *
     * @param key The label.
     * @return The int id for that label.
     */
    private int getIDOrThrow(Label key) {
        int id = domain.getID(key);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown label: " + key);
        }
        return id;
    }

    /**
     * Sets the label order used in {@link #toString}.
     * <p>
     * If the label order is a subset of the labels in the domain, only the
     * labels present in the label order will be displayed.
     *
     * @param newLabelOrder The label order to use.
     */
    @Override
    public void setLabelOrder(List<Label> newLabelOrder) {
        if (newLabelOrder == null || newLabelOrder.isEmpty()) {
            throw new IllegalArgumentException("Label order must be non-null and non-empty.");
        }
        this.labelOrder = Collections.unmodifiableList(new ArrayList<>(newLabelOrder));
    }

    /**
     * Gets the current label order.
     *
     * May trigger order instantiation if the label order has not been set.
     * @return The label order.
     */
    public List<Label> getLabelOrder() {
        return labelOrder;
    }

    @Override
    public String toString() {
        List<Label> curOrder = new ArrayList<>(labelOrder);
        curOrder.retainAll(observed);

        int maxLen = Integer.MIN_VALUE;
        for (Label label : curOrder) {
            maxLen = Math.max(label.getLabel().length(), maxLen);
            maxLen = Math.max(String.format(" %,d", (int)(double)occurrences.getOrDefault(label,0.0)).length(), maxLen);
        }

        StringBuilder sb = new StringBuilder();
        String trueLabelFormat = String.format("%%-%ds", maxLen + 2);
        String predictedLabelFormat = String.format("%%%ds", maxLen + 2);
        String countFormat = String.format("%%,%dd", maxLen + 2);

        //
        // Empty spot in first row for labels on subsequent rows.
        sb.append(String.format(trueLabelFormat, ""));

        //
        // Labels across the top for predicted.
        for (Label predictedLabel : curOrder) {
            sb.append(String.format(predictedLabelFormat, predictedLabel.getLabel()));
        }
        sb.append('\n');

        for (Label trueLabel : curOrder) {
            sb.append(String.format(trueLabelFormat, trueLabel.getLabel()));
            for (Label predictedLabel : curOrder) {
            	int confusion = (int) confusion(predictedLabel, trueLabel);
                sb.append(String.format(countFormat, confusion));
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    /**
     * Emits a HTML table representation of the Confusion Matrix.
     * @return The confusion matrix as a HTML table.
     */
    public String toHTML() {
        Set<Label> labelsToPrint = new LinkedHashSet<>(labelOrder);
        labelsToPrint.retainAll(observed);
        StringBuilder sb = new StringBuilder();
        sb.append("<table>\n");
        sb.append(String.format("<tr><th>True Label</th><th style=\"text-align:center\" colspan=\"%d\">Predicted Labels</th></tr>%n", occurrences.size() + 1));
        sb.append("<tr><th></th>");
        for (Label predictedLabel : labelsToPrint) {
            sb.append("<th style=\"text-align:right\">")
                    .append(predictedLabel)
                    .append("</th>");
        }
        sb.append("<th style=\"text-align:right\">Total</th>");
        sb.append("</tr>\n");
        for (Label trueLabel : labelsToPrint) {
            sb.append("<tr><th>").append(trueLabel).append("</th>");
            double count = occurrences.getOrDefault(trueLabel, 0d);
            for (Label predictedLabel : labelsToPrint) {
                double tlmc = confusion(predictedLabel,trueLabel);
                double percent = (tlmc / count) * 100;
                sb.append("<td style=\"text-align:right\">")
                        .append(String.format("%,d (%.1f%%)", (int)tlmc, percent))
                        .append("</td>");
            }
            sb.append("<td style=\"text-align:right\">").append(count).append("</td>");
            sb.append("</tr>\n");
        }
        sb.append("</table>");
        return sb.toString();
    }
}