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

package org.tribuo.classification.evaluation;

import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.evaluation.metrics.EvaluationMetric.Average;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.provenance.EvaluationProvenance;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * The implementation of {@link LabelEvaluation} for multi-class classification tasks.
 */
final class LabelEvaluationImpl implements LabelEvaluation {

    private final Map<MetricID<Label>, Double> results;
    private final LabelMetric.Context context;
    private final boolean modelGeneratesProbabilities;
    private final ConfusionMatrix<Label> cm;
    private final EvaluationProvenance provenance;

    LabelEvaluationImpl(Map<MetricID<Label>, Double> results,
                        LabelMetric.Context context,
                        EvaluationProvenance provenance) {
        this.results = results;
        this.context = context;
        this.provenance = provenance;
        this.modelGeneratesProbabilities = context.getModel().generatesProbabilities();
        this.cm = context.getCM();
    }

    @Override
    public List<Prediction<Label>> getPredictions() {
        return context.getPredictions();
    }

    @Override
    public Map<MetricID<Label>, Double> asMap() {
        return Collections.unmodifiableMap(results);
    }

    @Override
    public double averagedPrecision(Label label) {
        if (!modelGeneratesProbabilities) {
            throw new UnsupportedOperationException("averaged precision score not available for models that do not generate probabilities");
        }
        return get(label, LabelMetrics.AVERAGED_PRECISION);
    }

    /**
     * Note: precision recall curve is not stored in the underlying map, so it won't show up in aggregation.
     * @param label The label to use.
     * @return A PRCurve for that label.
     */
    @Override
    public LabelEvaluationUtil.PRCurve precisionRecallCurve(Label label) {
        return LabelMetrics.precisionRecallCurve(label, context.getPredictions());
    }

    @Override
    public double AUCROC(Label label) {
        if (!modelGeneratesProbabilities) {
            throw new UnsupportedOperationException("AUCROC score not available for models that do not generate probabilities");
        }
        return get(label, LabelMetrics.AUCROC);
    }

    /**
     * Note: averageAUCROC is not stored in the underlying map, so it won't show up in aggregation.
     * @param weighted If true weight by the class counts, if false use a macro average.
     * @return The average AUCROC.
     */
    @Override
    public double averageAUCROC(boolean weighted) {
        if (!modelGeneratesProbabilities) {
            throw new UnsupportedOperationException("AUCROC score not available for models that do not generate probabilities");
        }
        double sum = 0.0;
        double weightSum = 0.0;
        for (Label l : cm.getDomain().getDomain()) {
            double currentValue = get(l, LabelMetrics.AUCROC);
            double currentWeight = weighted ? cm.support(l) : 1.0;
            sum += currentWeight * currentValue;
            weightSum += currentWeight;
        }
        return sum / weightSum;
    }

    /**
     * Note: confusion is not stored in the underlying map, so it won't show up in aggregation.
     * @param predictedLabel The predicted label.
     * @param trueLabel The true label.
     * @return The number of times that {@code predictedLabel} was predicted for <code>trueLabel</code>.
     */
    @Override
    public double confusion(Label predictedLabel, Label trueLabel) {
        return cm.confusion(predictedLabel, trueLabel);
    }

    @Override
    public double tp(Label label) {
        return get(label, LabelMetrics.TP);
    }

    @Override
    public double tp() {
        return get(Average.MICRO, LabelMetrics.TP);
    }

    @Override
    public double macroTP() {
        return get(Average.MACRO, LabelMetrics.TP);
    }

    @Override
    public double fp(Label label) {
        return get(label, LabelMetrics.FP);
    }

    @Override
    public double fp() {
        return get(Average.MICRO, LabelMetrics.FP);
    }

    @Override
    public double macroFP() {
        return get(Average.MACRO, LabelMetrics.FP);
    }

    @Override
    public double tn(Label label) {
        return get(label, LabelMetrics.TN);
    }

    @Override
    public double tn() {
        return get(Average.MICRO, LabelMetrics.TN);
    }

    @Override
    public double macroTN() {
        return get(Average.MACRO, LabelMetrics.TN);
    }

    @Override
    public double fn(Label label) {
        return get(label, LabelMetrics.FN);
    }

    @Override
    public double fn() {
        return get(Average.MICRO, LabelMetrics.FN);
    }

    @Override
    public double macroFN() {
        return get(Average.MACRO, LabelMetrics.FN);
    }

    @Override
    public double precision(Label label) {
        return get(label, LabelMetrics.PRECISION);
    }

    @Override
    public double microAveragedPrecision() {
        return get(Average.MICRO, LabelMetrics.PRECISION);
    }

    @Override
    public double macroAveragedPrecision() {
        return get(Average.MACRO, LabelMetrics.PRECISION);
    }

    @Override
    public double recall(Label label) {
        return get(label, LabelMetrics.RECALL);
    }

    @Override
    public double microAveragedRecall() {
        return get(Average.MICRO, LabelMetrics.RECALL);
    }

    @Override
    public double macroAveragedRecall() {
        return get(Average.MACRO, LabelMetrics.RECALL);
    }

    @Override
    public double f1(Label label) {
        return get(label, LabelMetrics.F1);
    }

    @Override
    public double microAveragedF1() {
        return get(Average.MICRO, LabelMetrics.F1);
    }

    @Override
    public double macroAveragedF1() {
        return get(Average.MACRO, LabelMetrics.F1);
    }

    @Override
    public double accuracy() {
        return get(Average.MICRO, LabelMetrics.ACCURACY);
    }

    @Override
    public double accuracy(Label label) {
        return get(label, LabelMetrics.ACCURACY);
    }

    @Override
    public double balancedErrorRate() {
        // Target doesn't matter for balanced error rate, so we just use Average.macro
        // as it's the macro averaged recall.
        return get(Average.MACRO, LabelMetrics.BALANCED_ERROR_RATE);
    }

    @Override
    public ConfusionMatrix<Label> getConfusionMatrix() {
        return cm;
    }

    @Override
    public EvaluationProvenance getProvenance() { return provenance; }

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
        return LabelEvaluation.toFormattedString(this);
    }

    private double get(MetricTarget<Label> tgt, LabelMetrics metric) {
        return get(metric.forTarget(tgt).getID());
    }

    private double get(Label label, LabelMetrics metric) {
        return get(metric
                .forTarget(new MetricTarget<>(label))
                .getID());
    }

    private double get(Average avg, LabelMetrics metric) {
        return get(metric
                .forTarget(new MetricTarget<>(avg))
                .getID());
    }
}