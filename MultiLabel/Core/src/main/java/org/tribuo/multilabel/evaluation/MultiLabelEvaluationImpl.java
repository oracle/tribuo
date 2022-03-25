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

import org.tribuo.Prediction;
import org.tribuo.classification.evaluation.ConfusionMatrix;
import org.tribuo.evaluation.metrics.EvaluationMetric.Average;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.provenance.EvaluationProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;


/**
 * The implementation of a {@link MultiLabelEvaluation} using the default metrics.
 * <p>
 * The classification metrics consider labels independently.
 */
public final class MultiLabelEvaluationImpl implements MultiLabelEvaluation {

    private final Map<MetricID<MultiLabel>, Double> results;
    private final MultiLabelMetric.Context context;
    private final ConfusionMatrix<MultiLabel> cm;
    private final EvaluationProvenance provenance;

    /**
     * Builds an evaluation using the supplied metric results, confusion matrix and evaluation provenance.
     * @param results The results.
     * @param context The context carrying the confusion matrix
     * @param provenance The evaluation provenance.
     */
    MultiLabelEvaluationImpl(Map<MetricID<MultiLabel>, Double> results,
                             MultiLabelMetric.Context context,
                             EvaluationProvenance provenance) {
        this.results = results;
        this.context = context;
        this.cm = context.getCM();
        this.provenance = provenance;
    }

    @Override
    public List<Prediction<MultiLabel>> getPredictions() {
        return context.getPredictions();
    }

    @Override
    public double balancedErrorRate() {
        // Target doesn't matter for balanced error rate, so we just use Average.macro
        // as it's the macro average of the recalls.
        MetricTarget<MultiLabel> dummy = MetricTarget.macroAverageTarget();
        return get(dummy, MultiLabelMetrics.BALANCED_ERROR_RATE);
    }

    @Override
    public ConfusionMatrix<MultiLabel> getConfusionMatrix() {
        return cm;
    }

    @Override
    public double confusion(MultiLabel predicted, MultiLabel truth) {
        return cm.confusion(predicted, truth);
    }

    @Override
    public double tp(MultiLabel label) {
        return get(label, MultiLabelMetrics.TP);
    }

    @Override
    public double tp() {
        return get(Average.MICRO, MultiLabelMetrics.TP);
    }

    @Override
    public double macroTP() {
        return get(Average.MACRO, MultiLabelMetrics.TP);
    }

    @Override
    public double fp(MultiLabel label) {
        return get(label, MultiLabelMetrics.FP);
    }

    @Override
    public double fp() {
        return get(Average.MICRO, MultiLabelMetrics.FP);
    }

    @Override
    public double macroFP() {
        return get(Average.MACRO, MultiLabelMetrics.FP);
    }

    @Override
    public double tn(MultiLabel label) {
        return get(label, MultiLabelMetrics.TN);
    }

    @Override
    public double tn() {
        return get(Average.MICRO, MultiLabelMetrics.TN);
    }

    @Override
    public double macroTN() {
        return get(Average.MACRO, MultiLabelMetrics.TN);
    }

    @Override
    public double fn(MultiLabel label) {
        return get(label, MultiLabelMetrics.FN);
    }

    @Override
    public double fn() {
        return get(Average.MICRO, MultiLabelMetrics.FN);
    }

    @Override
    public double macroFN() {
        return get(Average.MACRO, MultiLabelMetrics.FN);
    }

    @Override
    public double precision(MultiLabel label) {
        return get(new MetricTarget<>(label), MultiLabelMetrics.PRECISION);
    }

    @Override
    public double microAveragedPrecision() {
        return get(new MetricTarget<>(Average.MICRO), MultiLabelMetrics.PRECISION);
    }

    @Override
    public double macroAveragedPrecision() {
        return get(new MetricTarget<>(Average.MACRO), MultiLabelMetrics.PRECISION);
    }

    @Override
    public double recall(MultiLabel label) {
        return get(new MetricTarget<>(label), MultiLabelMetrics.RECALL);
    }

    @Override
    public double microAveragedRecall() {
        return get(new MetricTarget<>(Average.MICRO), MultiLabelMetrics.RECALL);
    }

    @Override
    public double macroAveragedRecall() {
        return get(new MetricTarget<>(Average.MACRO), MultiLabelMetrics.RECALL);
    }

    @Override
    public double f1(MultiLabel label) {
        return get(new MetricTarget<>(label), MultiLabelMetrics.F1);
    }

    @Override
    public double microAveragedF1() {
        return get(new MetricTarget<>(Average.MICRO), MultiLabelMetrics.F1);
    }

    @Override
    public double macroAveragedF1() {
        return get(new MetricTarget<>(Average.MACRO), MultiLabelMetrics.F1);
    }

    @Override
    public double jaccardScore() {
        return get(new MetricTarget<>(Average.MACRO), MultiLabelMetrics.JACCARD_SCORE);
    }

    @Override
    public Map<MetricID<MultiLabel>, Double> asMap() {
        return Collections.unmodifiableMap(results);
    }

    @Override
    public EvaluationProvenance getProvenance() {
        return provenance;
    }

    /**
     * This method produces a nicely formatted String output, with
     * appropriate tabs and newlines, suitable for display on a terminal.
     * <p>
     * Uses the label order of the confusion matrix, which can be used to display
     * a subset of the per label metrics. When they are subset the total row
     * represents only the subset selected, not all the predictions, however
     * the accuracy and averaged metrics cover all the predictions.
     * @return Formatted output showing the main results of the evaluation.
     */
    @Override
    public String toString() {
        return toString(cm.getLabelOrder());
    }

    /**
     * This method produces a nicely formatted String output, with
     * appropriate tabs and newlines, suitable for display on a terminal.
     * <p>
     * Uses the label order of the confusion matrix, which can be used to display
     * a subset of the per label metrics. When they are subset the total row
     * represents only the subset selected, not all the predictions, however
     * the accuracy and averaged metrics cover all the predictions.
     *
     * @param labelOrder The label order to use.
     * @return Formatted output showing the main results of the evaluation.
     */
    private String toString(List<MultiLabel> labelOrder) {
        List<MultiLabel> retainedLabelOrder = new ArrayList<>(labelOrder);
        retainedLabelOrder.retainAll(cm.observed());
        StringBuilder sb = new StringBuilder();
        int tp = 0;
        int fn = 0;
        int fp = 0;
        int n = 0;
        //
        // Figure out the biggest class label and therefore the format string
        // that we should use for them.
        int maxLabelSize = "Balanced Error Rate".length();
        for(MultiLabel label : retainedLabelOrder) {
            maxLabelSize = Math.max(maxLabelSize, label.getLabelString().length());
        }
        String labelFormatString = String.format("%%-%ds", maxLabelSize+2);
        sb.append(String.format(labelFormatString, "Class"));
        sb.append(String.format("%12s%12s%12s%12s", "n", "tp", "fn", "fp"));
        sb.append(String.format("%12s%12s%12s%n", "recall", "prec", "f1"));
        for (MultiLabel label : retainedLabelOrder) {
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
            sb.append(String.format("%12.3f%12.3f%12.3f%n", recall(label), precision(label), f1(label)));
        }
        sb.append(String.format(labelFormatString, "Total"));
        sb.append(String.format("%,12d%,12d%,12d%,12d%n", n, tp, fn, fp));
        sb.append(String.format(labelFormatString, "Accuracy"));
        sb.append(String.format("%60.3f%n", cm.tp() / cm.support()));
        sb.append(String.format(labelFormatString, "Micro Average"));
        sb.append(String.format("%60.3f%12.3f%12.3f%n", microAveragedRecall(), microAveragedPrecision(), microAveragedF1()));
        sb.append(String.format(labelFormatString, "Macro Average"));
        sb.append(String.format("%60.3f%12.3f%12.3f%n", macroAveragedRecall(), macroAveragedPrecision(), macroAveragedF1()));
        sb.append(String.format(labelFormatString, "Balanced Error Rate"));
        sb.append(String.format("%60.3f%n", balancedErrorRate()));
        sb.append(String.format(labelFormatString, "Jaccard Score"));
        sb.append(String.format("%60.3f", jaccardScore()));
        return sb.toString();
    }

    private double get(MetricTarget<MultiLabel> tgt, MultiLabelMetrics metric) {
        return get(metric.forTarget(tgt).getID());
    }

    private double get(MultiLabel label, MultiLabelMetrics metric) {
        return get(metric
                .forTarget(new MetricTarget<>(label))
                .getID());
    }

    private double get(Average avg, MultiLabelMetrics metric) {
        return get(metric
                .forTarget(new MetricTarget<>(avg))
                .getID());
    }

    @Override
    public double get(MetricID<MultiLabel> key) {
        Double value = results.get(key);
        if (value == null) {
            throw new IllegalArgumentException("Metric value not found: " + key.toString());
        }
        return value;
    }

}