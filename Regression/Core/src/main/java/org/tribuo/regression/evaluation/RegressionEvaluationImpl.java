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

package org.tribuo.regression.evaluation;

import org.tribuo.Prediction;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.provenance.EvaluationProvenance;
import org.tribuo.regression.Regressor;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * The implementation of {@link RegressionEvaluation} using the default metrics.
 */
final class RegressionEvaluationImpl implements RegressionEvaluation {

    private final Map<MetricID<Regressor>, Double> results;
    private final RegressionMetric.Context context;
    private final RegressionSufficientStatistics memo;
    private final EvaluationProvenance provenance;

    RegressionEvaluationImpl(Map<MetricID<Regressor>, Double> results,
                             RegressionMetric.Context context,
                             EvaluationProvenance provenance) {
        this.results = results;
        this.context = context;
        this.memo = context.getMemo();
        this.provenance = provenance;
    }

    @Override
    public List<Prediction<Regressor>> getPredictions() {
        return context.getPredictions();
    }

    @Override
    public double averageMAE() {
        return get(MetricTarget.macroAverageTarget(), RegressionMetrics.MAE);
    }

    @Override
    public double mae(Regressor variable) {
        return get(new MetricTarget<>(variable), RegressionMetrics.MAE);
    }

    @Override
    public Map<Regressor, Double> mae() {
        return get(RegressionMetrics.MAE);
    }

    @Override
    public double averageR2() {
        return get(MetricTarget.macroAverageTarget(), RegressionMetrics.R2);
    }

    @Override
    public double r2(Regressor variable) {
        return get(new MetricTarget<>(variable), RegressionMetrics.R2);
    }

    @Override
    public Map<Regressor, Double> r2() {
        return get(RegressionMetrics.R2);
    }

    @Override
    public double averageRMSE() {
        return get(MetricTarget.macroAverageTarget(), RegressionMetrics.RMSE);
    }

    @Override
    public double rmse(Regressor variable) {
        return get(new MetricTarget<>(variable), RegressionMetrics.RMSE);
    }

    @Override
    public Map<Regressor, Double> rmse() {
        return get(RegressionMetrics.RMSE);
    }

    @Override
    public double averagedExplainedVariance() {
        return get(MetricTarget.macroAverageTarget(), RegressionMetrics.EV);
    }

    @Override
    public double explainedVariance(Regressor variable) {
        return get(new MetricTarget<>(variable), RegressionMetrics.EV);
    }

    @Override
    public Map<Regressor, Double> explainedVariance() {
        return get(RegressionMetrics.EV);
    }

    @Override
    public Map<MetricID<Regressor>, Double> asMap() {
        return Collections.unmodifiableMap(results);
    }

    @Override
    public EvaluationProvenance getProvenance() {
        return provenance;
    }

    @Override
    public String toString() {
        return "Multi-dimensional Regression Evaluation\nRMSE = " + convertKeys(rmse()) + "\nMean Absolute Error = " + convertKeys(mae()) +
                "\nR^2 = " + convertKeys(r2()) + "\nexplained variance = " + convertKeys(explainedVariance());
    }

    /**
     * The default toString on DimensionTuple emits the regressors minimum value.
     * This tidies it up by using the name as the key.
     * @param map The map to tidy.
     * @return A map with the dimension names as keys.
     */
    private static Map<String, Double> convertKeys(Map<Regressor, Double> map) {
        Map<String, Double> outputMap = new HashMap<>(map.size());

        for (Map.Entry<Regressor, Double> e : map.entrySet()) {
            outputMap.put(e.getKey().getDimensionNamesString(),e.getValue());
        }

        return outputMap;
    }

    private double get(MetricTarget<Regressor> target, RegressionMetrics metric) {
        return get(metric.forTarget(target).getID());
    }

    private Map<Regressor, Double> get(RegressionMetrics metric) {
        Map<Regressor, Double> map = new HashMap<>();
        for (Regressor variable : memo.domain.getDomain()) {
            MetricTarget<Regressor> target = new MetricTarget<>(variable);
            map.put(variable, get(target, metric));
        }
        return map;
    }
}