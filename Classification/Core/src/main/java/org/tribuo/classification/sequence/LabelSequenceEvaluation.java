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

package org.tribuo.classification.sequence;

import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.ConfusionMatrix;
import org.tribuo.classification.evaluation.LabelMetric;
import org.tribuo.classification.evaluation.LabelMetrics;
import org.tribuo.evaluation.metrics.EvaluationMetric;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.provenance.EvaluationProvenance;
import org.tribuo.sequence.SequenceEvaluation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * A class that can be used to evaluate a sequence label classification model element wise on a given set of data.
 */
public class LabelSequenceEvaluation implements SequenceEvaluation<Label> {

    private static final Logger logger = Logger.getLogger(LabelSequenceEvaluation.class.getName());
    
    private final Map<MetricID<Label>, Double> results;
    private final LabelMetric.Context ctx;
    private final ConfusionMatrix<Label> cm;
    private final EvaluationProvenance provenance;

    protected LabelSequenceEvaluation(Map<MetricID<Label>, Double> results,
                                      LabelMetric.Context ctx,
                                      EvaluationProvenance provenance) {
        this.results = results;
        this.ctx = ctx;
        this.cm = ctx.getCM();
        this.provenance = provenance;
    }

    /**
     * Gets the flattened predictions.
     * @return The flattened predictions.
     */
    public List<Prediction<Label>> getPredictions() {
        return ctx.getPredictions();
    }

    /**
     * Gets the confusion matrix backing this evaluation.
     * @return The confusion matrix.
     */
    public ConfusionMatrix<Label> getConfusionMatrix() {
        return cm;
    }

    @Override
    public Map<MetricID<Label>, Double> asMap() {
        return Collections.unmodifiableMap(results);
    }

    /**
     * Note: confusion is not stored in the underlying map, so it won't show up in aggregation.
     * @param predictedLabel The predicted label.
     * @param trueLabel The true label.
     * @return The number of times that {@code predictedLabel} was predicted for <code>trueLabel</code>.
     */
    public double confusion(Label predictedLabel, Label trueLabel) {
        return cm.confusion(predictedLabel, trueLabel);
    }

    public double tp(Label label) {
        return get(label, LabelMetrics.TP);
    }

    public double tp() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.TP);
    }

    public double macroTP() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.TP);
    }

    public double fp(Label label) {
        return get(label, LabelMetrics.FP);
    }

    public double fp() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.FP);
    }

    public double macroFP() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.FP);
    }

    public double tn(Label label) {
        return get(label, LabelMetrics.TN);
    }

    public double tn() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.TN);
    }

    public double macroTN() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.TN);
    }

    public double fn(Label label) {
        return get(label, LabelMetrics.FN);
    }

    public double fn() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.FN);
    }

    public double macroFN() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.FN);
    }

    public double precision(Label label) {
        return get(label, LabelMetrics.PRECISION);
    }

    public double microAveragedPrecision() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.PRECISION);
    }

    public double macroAveragedPrecision() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.PRECISION);
    }

    public double recall(Label label) {
        return get(label, LabelMetrics.RECALL);
    }

    public double microAveragedRecall() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.RECALL);
    }

    public double macroAveragedRecall() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.RECALL);
    }

    public double f1(Label label) {
        return get(label, LabelMetrics.RECALL);
    }

    public double microAveragedF1() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.F1);
    }

    public double macroAveragedF1() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.F1);
    }

    public double accuracy() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.ACCURACY);
    }

    public double accuracy(Label label) {
        return get(label, LabelMetrics.ACCURACY);
    }

    public double balancedErrorRate() {
        // Target doesn't matter for balanced error rate, so we just use Average.macro
        // as it's the macro averaged recall.
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.BALANCED_ERROR_RATE);
    }

    @Override
    public EvaluationProvenance getProvenance() { return provenance; }

    @Override
    public String toString() {
        List<Label> labelOrder = new ArrayList<>(cm.getDomain().getDomain());
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
            sb.append(String.format("%12.3f%12.3f%12.3f%n", recall(label), precision(label), f1(label)));
        }
        sb.append(String.format(labelFormatString, "Total"));
        sb.append(String.format("%,12d%,12d%,12d%,12d%n", n, tp, fn, fp));
        sb.append(String.format(labelFormatString, "Accuracy"));
        sb.append(String.format("%60.3f%n", (double) tp / n));
        sb.append(String.format(labelFormatString, "Micro Average"));
        sb.append(String.format("%60.3f%12.3f%12.3f%n", microAveragedRecall(), microAveragedPrecision(), microAveragedF1()));
        sb.append(String.format(labelFormatString, "Macro Average"));
        sb.append(String.format("%60.3f%12.3f%12.3f%n", macroAveragedRecall(), macroAveragedPrecision(), macroAveragedF1()));
        sb.append(String.format(labelFormatString, "Balanced Error Rate"));
        sb.append(String.format("%60.3f", balancedErrorRate()));
        return sb.toString();
    }

    private double get(MetricTarget<Label> tgt, LabelMetrics metric) {
        return get(metric.forTarget(tgt).getID());
    }

    private double get(Label label, LabelMetrics metric) {
        return get(metric
                .forTarget(new MetricTarget<>(label))
                .getID());
    }

    private double get(EvaluationMetric.Average avg, LabelMetrics metric) {
        return get(metric
                .forTarget(new MetricTarget<>(avg))
                .getID());
    }
}
