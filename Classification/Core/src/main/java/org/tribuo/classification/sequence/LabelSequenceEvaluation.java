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

    /**
     * Constructs a LabelSequenceEvaluation using the supplied parameters.
     * @param results The metric values.
     * @param ctx The context.
     * @param provenance The evaluation provenance.
     */
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

    /**
     * Gets the true positive count for that label.
     * @param label The label.
     * @return The true positive count.
     */
    public double tp(Label label) {
        return get(label, LabelMetrics.TP);
    }

    /**
     * Gets the micro averaged true positive count.
     * @return The micro averaged true positive count.
     */
    public double tp() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.TP);
    }

    /**
     * Gets the macro averaged true positive count.
     * @return The macro averaged true positive count.
     */
    public double macroTP() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.TP);
    }

    /**
     * The false positive count for this label.
     * @param label The label.
     * @return The false positive count.
     */
    public double fp(Label label) {
        return get(label, LabelMetrics.FP);
    }

    /**
     * Gets the micro averaged false positive count.
     * @return The micro averaged false positive count.
     */
    public double fp() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.FP);
    }

    /**
     * Gets the macro averaged false positive count.
     * @return The macro averaged false positive count.
     */
    public double macroFP() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.FP);
    }

    /**
     * The true negative count for this label.
     * @param label The label.
     * @return The true negative count.
     */
    public double tn(Label label) {
        return get(label, LabelMetrics.TN);
    }

    /**
     * Gets the micro averaged true negative count.
     * @return The micro averaged true negative count.
     */
    public double tn() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.TN);
    }

    /**
     * Gets the macro averaged true negative count.
     * @return The macro averaged true negative count.
     */
    public double macroTN() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.TN);
    }

    /**
     * The false negative count for this label.
     * @param label The label.
     * @return The false negative count.
     */
    public double fn(Label label) {
        return get(label, LabelMetrics.FN);
    }

    /**
     * Gets the micro averaged false negative count.
     * @return The micro averaged false negative count.
     */
    public double fn() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.FN);
    }

    /**
     * Gets the macro averaged false negative count.
     * @return The macro averaged false negative count.
     */
    public double macroFN() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.FN);
    }

    /**
     * The precision for this label.
     * @param label The label.
     * @return The precision.
     */
    public double precision(Label label) {
        return get(label, LabelMetrics.PRECISION);
    }

    /**
     * The micro averaged precision.
     * @return The micro averaged precision.
     */
    public double microAveragedPrecision() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.PRECISION);
    }

    /**
     * The macro averaged precision.
     * @return The macro averaged precision.
     */
    public double macroAveragedPrecision() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.PRECISION);
    }

    /**
     * The recall for this label.
     * @param label The label.
     * @return The recall.
     */
    public double recall(Label label) {
        return get(label, LabelMetrics.RECALL);
    }

    /**
     * The micro averaged recall.
     * @return The micro averaged recall.
     */
    public double microAveragedRecall() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.RECALL);
    }

    /**
     * The macro averaged recall.
     * @return The macro averaged recall.
     */
    public double macroAveragedRecall() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.RECALL);
    }

    /**
     * The F1 for this label.
     * @param label The label.
     * @return The F1.
     */
    public double f1(Label label) {
        return get(label, LabelMetrics.RECALL);
    }

    /**
     * The micro averaged F1.
     * @return The micro averaged F1.
     */
    public double microAveragedF1() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.F1);
    }

    /**
     * The macro averaged F1.
     * @return The macro averaged F1.
     */
    public double macroAveragedF1() {
        return get(EvaluationMetric.Average.MACRO, LabelMetrics.F1);
    }

    /**
     * The accuracy.
     * @return The accuracy.
     */
    public double accuracy() {
        return get(EvaluationMetric.Average.MICRO, LabelMetrics.ACCURACY);
    }

    /**
     * Gets the accuracy for this label.
     * @param label The label.
     * @return The accuracy.
     */
    public double accuracy(Label label) {
        return get(label, LabelMetrics.ACCURACY);
    }

    /**
     * Gets the balanced error rate.
     * <p>
     * Also known as 1 - the macro averaged recall.
     * @return The balanced error rate.
     */
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
