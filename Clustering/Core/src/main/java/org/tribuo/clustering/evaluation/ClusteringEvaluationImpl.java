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

package org.tribuo.clustering.evaluation;

import org.tribuo.Prediction;
import org.tribuo.clustering.ClusterID;
import org.tribuo.evaluation.metrics.EvaluationMetric.Average;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.provenance.EvaluationProvenance;

import java.util.Collections;
import java.util.List;
import java.util.Map;


/**
 * The implementation of {@link ClusteringEvaluation}.
 */
final class ClusteringEvaluationImpl implements ClusteringEvaluation {

    private final Map<MetricID<ClusterID>, Double> results;
    private final ClusteringMetric.Context context;
    private final EvaluationProvenance provenance;

    /**
     * Builds an evaluation from the metrics and provenance.
     * @param results The metric results.
     * @param provenance The evaluation provenance.
     */
    ClusteringEvaluationImpl(Map<MetricID<ClusterID>, Double> results,
         ClusteringMetric.Context context,
         EvaluationProvenance provenance) {
        this.results = results;
        this.context = context;
        this.provenance = provenance;
    }

    @Override
    public List<Prediction<ClusterID>> getPredictions() {
        return context.getPredictions();
    }

    @Override
    public double normalizedMI() {
        // Just using Average.micro here arbitrarily, NMI/AMI don't actually need targets
        MetricTarget<ClusterID> target = new MetricTarget<>(Average.MICRO);
        return get(target, ClusteringMetrics.NORMALIZED_MI);
    }

    @Override
    public double adjustedMI() {
        // Just using Average.micro here arbitrarily, NMI/AMI don't actually need targets
        MetricTarget<ClusterID> target = new MetricTarget<>(Average.MICRO);
        return get(target, ClusteringMetrics.ADJUSTED_MI);
    }

    @Override
    public Map<MetricID<ClusterID>, Double> asMap() {
        return Collections.unmodifiableMap(results);
    }

    @Override
    public EvaluationProvenance getProvenance() {
        return provenance;
    }

    @Override
    public String toString() {
        return "Clustering Evaluation\nNormalized MI = " + normalizedMI() + "\n" + "Adjusted MI = " + adjustedMI();
    }

    private double get(MetricTarget<ClusterID> target, ClusteringMetrics metric) {
        return get(metric.forTarget(target).getID());
    }
}