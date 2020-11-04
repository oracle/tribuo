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

package org.tribuo.anomaly.evaluation;

import org.tribuo.anomaly.AnomalyFactory;
import org.tribuo.anomaly.Event;
import org.tribuo.evaluation.metrics.MetricTarget;

import java.util.function.ToDoubleBiFunction;

/**
 * Default metrics for evaluating anomaly detection.
 */
public enum AnomalyMetrics {
    //
    // cast to doubles here because AbstractEvaluator assumes all metric results are doubles.
    /**
     * The number of true positives.
     */
    TP((t, c) -> (double) c.getTruePositive()),
    /**
     * The number of false positives.
     */
    FP((t, c) -> (double) c.getFalsePositive()),
    /**
     * The number of true negatives.
     */
    TN((t, c) -> (double) c.getTrueNegative()),
    /**
     * The number of false negatives.
     */
    FN((t, c) -> (double) c.getFalseNegative()),
    /**
     * The precision, i.e., the true positives divided by the predicted positives.
     */
    PRECISION((t,c) -> ((double) c.getTruePositive()) / (c.getTruePositive() + c.getFalsePositive())),
    /**
     * The recall, i.e., the true positives divided by the ground truth positives.
     */
    RECALL((t,c) -> ((double) c.getTruePositive()) / (c.getTruePositive() + c.getFalseNegative())),
    /**
     * The F_1 score, i.e., the harmonic mean of the precision and the recall.
     */
    F1((t,c) -> (2.0 * c.getTruePositive()) / ((2.0 * c.getTruePositive()) + c.getFalseNegative() + c.getFalsePositive()));

    private final ToDoubleBiFunction<MetricTarget<Event>, AnomalyMetric.Context> impl;

    private final AnomalyMetric metric;

    AnomalyMetrics(ToDoubleBiFunction<MetricTarget<Event>, AnomalyMetric.Context> impl) {
        this.impl = impl;
        this.metric = new AnomalyMetric(new MetricTarget<>(AnomalyFactory.ANOMALOUS_EVENT),this.name(),this.impl);
    }

    AnomalyMetric asMetric() {
        return metric;
    }
}