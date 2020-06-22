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

import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.anomaly.Event;
import org.tribuo.evaluation.AbstractEvaluator;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.provenance.EvaluationProvenance;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * An {@link Evaluator} for anomaly detection {@link Event}s.
 */
public class AnomalyEvaluator extends AbstractEvaluator<Event, AnomalyMetric.Context, AnomalyEvaluation, AnomalyMetric> {

    @Override
    protected Set<AnomalyMetric> createMetrics(Model<Event> model) {
        Set<AnomalyMetric> metrics = new HashSet<>();
        metrics.add(AnomalyMetrics.TP.asMetric());
        metrics.add(AnomalyMetrics.FP.asMetric());
        metrics.add(AnomalyMetrics.TN.asMetric());
        metrics.add(AnomalyMetrics.FN.asMetric());
        metrics.add(AnomalyMetrics.PRECISION.asMetric());
        metrics.add(AnomalyMetrics.RECALL.asMetric());
        metrics.add(AnomalyMetrics.F1.asMetric());
        return metrics;
    }

    @Override
    protected AnomalyMetric.Context createContext(Model<Event> model, List<Prediction<Event>> predictions) {
        return AnomalyMetric.buildContext(model, predictions);
    }

    @Override
    protected AnomalyEvaluation createEvaluation(AnomalyMetric.Context context,
                                                 Map<MetricID<Event>, Double> results,
                                                 EvaluationProvenance provenance) {
        return new AnomalyEvaluationImpl(results, context, provenance);
    }

}