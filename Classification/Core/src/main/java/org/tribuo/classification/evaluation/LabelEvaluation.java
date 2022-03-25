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

import org.tribuo.classification.Label;
import org.tribuo.evaluation.EvaluationRenderer;

import java.util.ArrayList;
import java.util.List;

/**
 * Adds multi-class classification specific metrics to {@link ClassifierEvaluation}.
 */
public interface LabelEvaluation extends ClassifierEvaluation<Label> {

    /**
     * The overall accuracy of the evaluation.
     * @return The accuracy.
     */
    double accuracy();

    /**
     * The per label accuracy of the evaluation.
     * @param label The target label.
     * @return The per label accuracy.
     */
    double accuracy(Label label);

    /**
     * Area under the ROC curve.
     *
     * @param label target label
     * @return AUC ROC score
     *
     * @implSpec Implementations of this class are expected to throw {@link UnsupportedOperationException} if the model
     * corresponding to this evaluation does not generate probabilities, which are required to compute the ROC curve.
     */
    double AUCROC(Label label);

    /**
     * Area under the ROC curve averaged across labels.
     * <p>
     * If {@code weighted} is false, use a macro average, if true, weight by the evaluation's observed class counts.
     * </p>
     *
     * @param weighted If true weight by the class counts, if false use a macro average.
     * @return The average AUCROC.
     *
     * @implSpec Implementations of this class are expected to throw {@link UnsupportedOperationException} if the model
     * corresponding to this evaluation does not generate probabilities, which are required to compute the ROC curve.
     */
    double averageAUCROC(boolean weighted);

    /**
     * Summarises a Precision-Recall Curve by taking the weighted mean of the
     * precisions at a given threshold, where the weight is the recall achieved at
     * that threshold.
     *
     * @see LabelEvaluationUtil#averagedPrecision(boolean[], double[])
     *
     * @param label The target label.
     * @return The averaged precision for that label.
     *
     * @implSpec Implementations of this class are expected to throw {@link UnsupportedOperationException} if the model
     * corresponding to this evaluation does not generate probabilities, which are required to compute the ROC curve.
     */
    double averagedPrecision(Label label);

    /**
     * Calculates the Precision Recall curve for a single label.
     *
     * @see LabelEvaluationUtil#generatePRCurve(boolean[], double[])
     *
     * @param label The target label.
     * @return The precision recall curve for that label.
     *
     * @implSpec Implementations of this class are expected to throw {@link UnsupportedOperationException} if the model
     * corresponding to this evaluation does not generate probabilities, which are required to compute the ROC curve.
     */
    LabelEvaluationUtil.PRCurve precisionRecallCurve(Label label);

    /**
     * Returns a HTML formatted String representing this evaluation.
     * <p>
     * Uses the label order of the confusion matrix, which can be used to display
     * a subset of the per label metrics. When they are subset the total row
     * represents only the subset selected, not all the predictions, however
     * the accuracy and averaged metrics cover all the predictions.
     * @return A HTML formatted String.
     */
    default String toHTML() {
        return LabelEvaluation.toHTML(this);
    }

    /**
     * This method produces a nicely formatted String output, with
     * appropriate tabs and newlines, suitable for display on a terminal.
     * It can be used as an implementation of the {@link EvaluationRenderer}
     * functional interface.
     * <p>
     * Uses the label order of the confusion matrix, which can be used to display
     * a subset of the per label metrics. When they are subset the total row
     * represents only the subset selected, not all the predictions, however
     * the accuracy and averaged metrics cover all the predictions.
     * @param evaluation The evaluation to format.
     * @return Formatted output showing the main results of the evaluation.
     */
    public static String toFormattedString(LabelEvaluation evaluation) {
        ConfusionMatrix<Label> cm = evaluation.getConfusionMatrix();
        List<Label> labelOrder = new ArrayList<>(cm.getLabelOrder());
        labelOrder.retainAll(cm.observed());
        StringBuilder sb = new StringBuilder();
        int tp = 0;
        int fn = 0;
        int fp = 0;
        int n = 0;
        //
        // Figure out the biggest class label and therefore the format string
        // that we should use for them.
        int maxLabelSize = "Balanced Error Rate".length();
        for(Label label : labelOrder) {
            maxLabelSize = Math.max(maxLabelSize, label.getLabel().length());
        }
        String labelFormatString = String.format("%%-%ds", maxLabelSize+2);
        sb.append(String.format(labelFormatString, "Class"));
        sb.append(String.format("%12s%12s%12s%12s", "n", "tp", "fn", "fp"));
        sb.append(String.format("%12s%12s%12s%n", "recall", "prec", "f1"));
        for (Label label : labelOrder) {
            if (cm.support(label) == 0) {
                continue;
            }
            n += cm.support(label);
            tp += cm.tp(label);
            fn += cm.fn(label);
            fp += cm.fp(label);
            sb.append(String.format(labelFormatString, label));
            sb.append(String.format("%,12d%,12d%,12d%,12d",
                    (int) cm.support(label),
                    (int) cm.tp(label),
                    (int) cm.fn(label),
                    (int) cm.fp(label)
            ));
            sb.append(String.format("%12.3f%12.3f%12.3f%n",
                    evaluation.recall(label),
                    evaluation.precision(label),
                    evaluation.f1(label)));
        }
        sb.append(String.format(labelFormatString, "Total"));
        sb.append(String.format("%,12d%,12d%,12d%,12d%n", n, tp, fn, fp));
        sb.append(String.format(labelFormatString, "Accuracy"));
        sb.append(String.format("%60.3f%n", evaluation.accuracy()));
        sb.append(String.format(labelFormatString, "Micro Average"));
        sb.append(String.format("%60.3f%12.3f%12.3f%n",
                evaluation.microAveragedRecall(),
                evaluation.microAveragedPrecision(),
                evaluation.microAveragedF1()));
        sb.append(String.format(labelFormatString, "Macro Average"));
        sb.append(String.format("%60.3f%12.3f%12.3f%n",
                evaluation.macroAveragedRecall(),
                evaluation.macroAveragedPrecision(),
                evaluation.macroAveragedF1()));
        sb.append(String.format(labelFormatString, "Balanced Error Rate"));
        sb.append(String.format("%60.3f", evaluation.balancedErrorRate()));
        return sb.toString();
    }

    /**
     * This method produces a HTML formatted String output, with
     * appropriate tabs and newlines, suitable for integration into a webpage.
     * It can be used as an implementation of the {@link EvaluationRenderer}
     * functional interface.
     * <p>
     * Uses the label order of the confusion matrix, which can be used to display
     * a subset of the per label metrics. When they are subset the total row
     * represents only the subset selected, not all the predictions, however
     * the accuracy and averaged metrics cover all the predictions.
     * @param evaluation The evaluation to format.
     * @return Formatted HTML output showing the main results of the evaluation.
     */
    public static String toHTML(LabelEvaluation evaluation) {
        ConfusionMatrix<Label> cm = evaluation.getConfusionMatrix();
        List<Label> labelOrder = cm.getLabelOrder();
        StringBuilder sb = new StringBuilder();
        int tp = 0;
        int fn = 0;
        int fp = 0;
        int tn = 0;
        sb.append("<table>\n");
        sb.append("<tr>\n");
        sb.append("<th>Class</th><th>n</th> <th>%</th> <th>tp</th> <th>fn</th> <th>fp</th> <th>Recall</th> <th>Precision</th> <th>F1</th>");
        sb.append("\n</tr>\n");
        //
        // Compute the total number of instances first, so we can show proportions.
        for (Label label : labelOrder) {
            //tn += occurrences.getOrDefault(label, 0);
            tn += cm.tn(label);
        }
        for (Label label : labelOrder) {
            if (cm.support(label) == 0) {
                continue;
            }
            tp += cm.tp(label);
            fn += cm.fn(label);
            fp += cm.fp(label);
            sb.append("<tr>");
            sb.append("<td><code>").append(label).append("</code></td>");
            int occurrence = (int) cm.support(label);
            sb.append("<td style=\"text-align:right\">").append(String.format("%,d", occurrence)).append("</td>");
            sb.append("<td style=\"text-align:right\">").append(String.format("%8.1f%%", (occurrence/ (double) tn)*100)).append("</td>");
            sb.append("<td style=\"text-align:right\">").append(String.format("%,d", (int) cm.tp(label))).append("</td>");
            sb.append("<td style=\"text-align:right\">").append(String.format("%,d", (int) cm.fn(label))).append("</td>");
            sb.append("<td style=\"text-align:right\">").append(String.format("%,d", (int) cm.fp(label))).append("</td>");
            sb.append(String.format("<td style=\"text-align:right\">%8.3f</td><td style=\"text-align:right\">%8.3f</td><td style=\"text-align:right\">%8.3f</td>%n",
                    evaluation.recall(label), evaluation.precision(label), evaluation.f1(label)));
            sb.append("</tr>");
        }
        sb.append("<tr>");
        sb.append("<td>Total</td>");
        sb.append(String.format("<td style=\"text-align:right\">%,12d</td><td style=\"text-align:right\"></td><td style=\"text-align:right\">%,12d</td><td style=\"text-align:right\">%,12d</td><td style=\"text-align:right\">%,12d</td>%n", tn, tp, fn, fp));
        sb.append("<td colspan=\"4\"></td>");
        sb.append("</tr>\n<tr>");
        sb.append(String.format("<td>Accuracy</td><td style=\"text-align:right\" colspan=\"6\">%8.3f</td>%n", evaluation.accuracy()));
        sb.append("<td colspan=\"4\"></td>");
        sb.append("</tr>\n<tr>");
        sb.append("<td>Micro Average</td>");
        sb.append(String.format("<td style=\"text-align:right\" colspan=\"6\">%8.3f</td><td style=\"text-align:right\">%8.3f</td><td style=\"text-align:right\">%8.3f</td>%n",
                evaluation.microAveragedRecall(),
                evaluation.microAveragedPrecision(),
                evaluation.microAveragedF1()));
        sb.append("</tr></table>");
        return sb.toString();
    }

}
