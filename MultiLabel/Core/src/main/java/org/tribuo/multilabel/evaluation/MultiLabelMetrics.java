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

import org.tribuo.classification.evaluation.ConfusionMetrics;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.multilabel.MultiLabel;

import java.util.function.BiFunction;

/**
 * An enum of the default {@link MultiLabelMetric}s supported by the multi-label classification
 * evaluation package.
 */
public enum MultiLabelMetrics {

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
     * The precision, i.e. the number of true positives divided by the number of predicted positives.
     */
    PRECISION((tgt, ctx) -> ConfusionMetrics.precision(tgt, ctx.getCM())),
    /**
     * The recall, i.e. the number of true positives divided by the number of ground truth positives.
     */
    RECALL((tgt, ctx) -> ConfusionMetrics.recall(tgt, ctx.getCM())),
    /**
     * The F_1 score, i.e. the harmonic mean of the precision and the recall.
     */
    F1((tgt, ctx) -> ConfusionMetrics.f1(tgt, ctx.getCM())),
    /**
     * The balanced error rate, i.e. the mean of the per class recalls.
     */
    BALANCED_ERROR_RATE((tgt, ctx) -> ConfusionMetrics.balancedErrorRate(ctx.getCM()));

    private final BiFunction<MetricTarget<MultiLabel>, MultiLabelMetric.Context, Double> impl;

    MultiLabelMetrics(BiFunction<MetricTarget<MultiLabel>, MultiLabelMetric.Context, Double> impl) {
        this.impl = impl;
    }

    public BiFunction<MetricTarget<MultiLabel>, MultiLabelMetric.Context, Double> getImpl() {
        return impl;
    }

    public MultiLabelMetric forTarget(MetricTarget<MultiLabel> tgt) {
        return new MultiLabelMetric(tgt, this.name(), this.getImpl());
    }
}