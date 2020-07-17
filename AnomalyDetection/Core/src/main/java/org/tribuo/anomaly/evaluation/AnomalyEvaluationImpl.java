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

import org.tribuo.Prediction;
import org.tribuo.anomaly.Event;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.provenance.EvaluationProvenance;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * The implementation of {@link AnomalyEvaluation}.
 */
final class AnomalyEvaluationImpl implements AnomalyEvaluation {

    private final Map<MetricID<Event>, Double> results;
    private final AnomalyMetric.Context context;
    private final EvaluationProvenance provenance;

    AnomalyEvaluationImpl(Map<MetricID<Event>, Double> results, AnomalyMetric.Context ctx, EvaluationProvenance provenance) {
        this.results = results;
        this.context = ctx;
        this.provenance = provenance;
    }

    @Override
    public List<Prediction<Event>> getPredictions() {
        return context.getPredictions();
    }

    @Override
    public long getFalsePositives() {
        return (long) get(AnomalyMetrics.FP);
    }

    @Override
    public long getTruePositives() {
        return (long) get(AnomalyMetrics.TP);
    }

    @Override
    public long getTrueNegatives() {
        return (long) get(AnomalyMetrics.TN);
    }

    @Override
    public long getFalseNegatives() {
        return (long) get(AnomalyMetrics.FN);
    }

    @Override
    public double getPrecision() {
        return get(AnomalyMetrics.PRECISION);
    }

    @Override
    public double getRecall() {
        return get(AnomalyMetrics.RECALL);
    }

    @Override
    public double getF1() {
        return get(AnomalyMetrics.F1);
    }

    @Override
    public Map<MetricID<Event>, Double> asMap() {
        return Collections.unmodifiableMap(results);
    }

    @Override
    public EvaluationProvenance getProvenance() {
        return provenance;
    }

    @Override
    public String toString() {
        return String.format("AnomalyEvaluation(tp=%d fp=%d tn=%d fn=%d precision=%f recall=%f f1=%f)",
                getTruePositives(), getFalsePositives(), getTrueNegatives(), getFalseNegatives(), getPrecision(), getRecall(), getF1());
    }

    @Override
    public String confusionString() {
        int maxLen = Integer.MIN_VALUE;
        maxLen = Math.max(Event.EventType.EXPECTED.name().length(), maxLen);
        maxLen = Math.max(Event.EventType.ANOMALOUS.name().length(), maxLen);
        maxLen = Math.max(String.format(" %,d", getTrueNegatives()).length(), maxLen);
        maxLen = Math.max(String.format(" %,d", getTruePositives()).length(), maxLen);
        maxLen = Math.max(String.format(" %,d", getFalseNegatives()).length(), maxLen);
        maxLen = Math.max(String.format(" %,d", getFalsePositives()).length(), maxLen);
        StringBuilder sb = new StringBuilder();
        String trueLabelFormat = String.format("%%-%ds", maxLen + 2);
        String predictedLabelFormat = String.format("%%%ds", maxLen + 2);
        String countFormat = String.format("%%,%dd", maxLen + 2);

        //
        // Empty spot in first row for labels on subsequent rows.
        sb.append(String.format(trueLabelFormat, ""));

        //
        // Labels across the top for predicted.
        sb.append(String.format(predictedLabelFormat, Event.EventType.EXPECTED.name()));
        sb.append(String.format(predictedLabelFormat, Event.EventType.ANOMALOUS.name()));
        sb.append('\n');

        //
        // First row, true label expected
        sb.append(String.format(trueLabelFormat, Event.EventType.EXPECTED.name()));
        sb.append(String.format(countFormat, getTrueNegatives()));
        sb.append(String.format(countFormat, getFalsePositives()));
        sb.append('\n');

        // Second row, true label anomalous
        sb.append(String.format(trueLabelFormat, Event.EventType.ANOMALOUS.name()));
        sb.append(String.format(countFormat, getFalseNegatives()));
        sb.append(String.format(countFormat, getTruePositives()));
        sb.append('\n');

        return sb.toString();
    }

    private double get(AnomalyMetrics metric) {
        double value = get(metric.asMetric().getID());
        return value;
    }
}