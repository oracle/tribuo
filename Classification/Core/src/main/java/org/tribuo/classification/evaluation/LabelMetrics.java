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

import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelMetric.Context;
import org.tribuo.evaluation.metrics.MetricTarget;

import java.util.List;
import java.util.function.ToDoubleBiFunction;

/**
 * An enum of the default {@link LabelMetric}s supported by the multi-class classification
 * evaluation package.
 */
public enum LabelMetrics {

    /**
     * The number of true positives.
     */
    TP((tgt, ctx) -> ConfusionMetrics.tp(tgt, ctx.getCM())),
    /**
     * The number of false positives.
     */
    FP((tgt, ctx) -> ConfusionMetrics.fp(tgt, ctx.getCM())),
    /**
     * The number of true negatives.
     */
    TN((tgt, ctx) -> ConfusionMetrics.tn(tgt, ctx.getCM())),
    /**
     * The number of false negatives.
     */
    FN((tgt, ctx) -> ConfusionMetrics.fn(tgt, ctx.getCM())),
    /**
     * The precision, i.e., the number of true positives divided by the number of predicted positives.
     */
    PRECISION((tgt, ctx) -> ConfusionMetrics.precision(tgt, ctx.getCM())),
    /**
     * The recall, i.e., the number of true positives divided by the number of ground truth positives.
     */
    RECALL((tgt, ctx) -> ConfusionMetrics.recall(tgt, ctx.getCM())),
    /**
     * The F_1 score, i.e., the harmonic mean of the precision and the recall.
     */
    F1((tgt, ctx) -> ConfusionMetrics.f1(tgt, ctx.getCM())),
    /**
     * The accuracy.
     */
    ACCURACY((tgt, ctx) -> ConfusionMetrics.accuracy(tgt, ctx.getCM())),
    /**
     * The balanced error rate, i.e., the mean of the per class recalls.
     */
    BALANCED_ERROR_RATE((tgt, ctx) -> ConfusionMetrics.balancedErrorRate(ctx.getCM())),
    /**
     * The area under the receiver-operator curve (ROC).
     */
    AUCROC((tgt, ctx) -> LabelMetrics.AUCROC(tgt, ctx.getPredictions())),
    /**
     * The averaged precision.
     */
    AVERAGED_PRECISION((tgt, ctx) -> LabelMetrics.averagedPrecision(tgt, ctx.getPredictions()));

    private final ToDoubleBiFunction<MetricTarget<Label>, LabelMetric.Context> impl;

    LabelMetrics(ToDoubleBiFunction<MetricTarget<Label>, LabelMetric.Context> impl) {
        this.impl = impl;
    }

    /**
     * Returns the implementing function for this metric.
     * @return The implementing function.
     */
    public ToDoubleBiFunction<MetricTarget<Label>, Context> getImpl() {
        return impl;
    }

    /**
     * Gets the LabelMetric wrapped around the supplied MetricTarget.
     * @param tgt The metric target.
     * @return The label metric combining the implementation function with the supplied metric target.
     */
    public LabelMetric forTarget(MetricTarget<Label> tgt) {
        return new LabelMetric(tgt, this.name(), this.getImpl());
    }

    /**
     * @see LabelEvaluationUtil#averagedPrecision(boolean[], double[])
     * @param tgt The metric target to use.
     * @param predictions The predictions to use.
     * @return The averaged precision for the supplied target with the supplied predictions.
     */
    public static double averagedPrecision(MetricTarget<Label> tgt, List<Prediction<Label>> predictions) {
        if (tgt.getOutputTarget().isPresent()) {
            return averagedPrecision(tgt.getOutputTarget().get(), predictions);
        } else {
            throw new IllegalStateException("Unsupported MetricTarget for averagedPrecision");
        }
    }

    /**
     * @see LabelEvaluationUtil#averagedPrecision(boolean[], double[])
     * @param label The Label to average across.
     * @param predictions The predictions to use.
     * @return The averaged precision for the supplied label with the supplied predictions.
     */
    public static double averagedPrecision(Label label, List<Prediction<Label>> predictions) {
        PredictionProbabilities record = new PredictionProbabilities(label, predictions);
        return LabelEvaluationUtil.averagedPrecision(record.ypos, record.yscore);
    }

    /**
     * @see LabelEvaluationUtil#generatePRCurve(boolean[], double[])
     * @param label The Label to calculate precision and recall for.
     * @param predictions The predictions to use.
     * @return The Precision Recall Curve for the supplied label with the supplied predictions.
     */
    public static LabelEvaluationUtil.PRCurve precisionRecallCurve(Label label, List<Prediction<Label>> predictions) {
        PredictionProbabilities record = new PredictionProbabilities(label, predictions);
        return LabelEvaluationUtil.generatePRCurve(record.ypos, record.yscore);
    }

    /**
     * Area under the ROC curve.
     *
     * @param label the label corresponding to the "positive" class
     * @param predictions the predictions for which we'll compute the score
     * @return AUC ROC score
     * @throws UnsupportedOperationException if a prediction with no probability score, which are required to compute the ROC curve. (See also: {@link Model#generatesProbabilities()})
     */
    public static double AUCROC(Label label, List<Prediction<Label>> predictions) {
        PredictionProbabilities record = new PredictionProbabilities(label, predictions);
        return LabelEvaluationUtil.binaryAUCROC(record.ypos, record.yscore);
    }

    /**
     * Area under the ROC curve.
     *
     * @param tgt The metric target for the positive class.
     * @param predictions the predictions for which we'll compute the score
     * @return AUC ROC score
     * @throws UnsupportedOperationException if a prediction with no probability score, which are required to compute the ROC curve. (See also: {@link Model#generatesProbabilities()})
     */
    public static double AUCROC(MetricTarget<Label> tgt, List<Prediction<Label>> predictions) {
        if (tgt.getOutputTarget().isPresent()) {
            return AUCROC(tgt.getOutputTarget().get(), predictions);
        } else {
            throw new IllegalStateException("Unsupported MetricTarget for AUCROC");
        }
    }

    /**
     * One day, it'll be a record. Not today mind.
     */
    private static final class PredictionProbabilities {
        final boolean[] ypos;
        final double[] yscore;
        PredictionProbabilities(Label label, List<Prediction<Label>> predictions) {
            int n = predictions.size();
            ypos = new boolean[n];
            yscore = new double[n];
            for (int i = 0; i < n; i++) {
                Prediction<Label> prediction = predictions.get(i);
                if (!prediction.hasProbabilities()) {
                    throw new UnsupportedOperationException(String.format("Invalid prediction at index %d: has no probability score.", i));
                }
                if (prediction.getExample().getOutput().equals(label)) {
                    ypos[i] = true;
                }
                yscore[i] = prediction
                        .getOutputScores()
                        .get(label.getLabel())
                        .getScore();
            }
        }
    }

}