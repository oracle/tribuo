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

import org.tribuo.evaluation.metrics.EvaluationMetric;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.regression.Regressor;
import org.tribuo.util.Util;

import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

/**
 * An enum of the default {@link RegressionMetric}s supported by the multi-dimensional regression
 * evaluation package.
 * <p>
 * The metrics treat each regressed dimension independently.
 */
public enum RegressionMetrics {

    /**
     * Calculates the R^2 of the predictions.
     */
    R2((target, context) -> RegressionMetrics.r2(target, context.getMemo())),
    /**
     * Calculates the Root Mean Squared Error of the predictions.
     */
    RMSE((target, context) -> RegressionMetrics.rmse(target, context.getMemo())),
    /**
     * Calculates the Mean Absolute Error of the predictions.
     */
    MAE((target, context) -> RegressionMetrics.mae(target, context.getMemo())),
    /**
     * Calculates the Explained Variance of the predictions.
     */
    EV((target, context) -> RegressionMetrics.explainedVariance(target, context.getMemo()));

    private final ToDoubleBiFunction<MetricTarget<Regressor>, RegressionMetric.Context> impl;
    RegressionMetrics(ToDoubleBiFunction<MetricTarget<Regressor>, RegressionMetric.Context> impl) {
        this.impl = impl;
    }

    RegressionMetric forTarget(MetricTarget<Regressor> target) {
        return new RegressionMetric(target, this.name(), this.impl);
    }

    /**
     * Calculates R^2 based on the supplied statistics.
     * @param target The regression dimension or average to target.
     * @param sufficientStats The sufficient statistics.
     * @return The R^2 value of the predictions.
     */
    public static double r2(MetricTarget<Regressor> target, RegressionSufficientStatistics sufficientStats) {
        return compute(target, sufficientStats, RegressionMetrics::r2);
    }

    /**
     * Calculates R^2 based on the supplied statistics for a single dimension.
     * @param variable The regression dimension.
     * @param sufficientStats The sufficient statistics.
     * @return The R^2 value of the predictions.
     */
    public static double r2(Regressor variable, RegressionSufficientStatistics sufficientStats) {
        String varname = variable.getNames()[0];
        double[] trueArray = sufficientStats.trueValues.get(varname);
        double numerator = sufficientStats.sumSquaredError.get(varname).doubleValue();
        double meanTruth = Util.weightedMean(trueArray, sufficientStats.weights, sufficientStats.n);
        double denominator = 0.0;
        for (int i = 0; i < sufficientStats.n; i++) {
            double difference = trueArray[i] - meanTruth;
            float currWeight = sufficientStats.weights[i];
            denominator += currWeight * difference * difference;
        }
        return 1.0 - (numerator / denominator);
    }

    /**
     * Calculates the RMSE based on the supplied statistics.
     * @param target The regression dimension or average to target.
     * @param sufficientStats The sufficient statistics.
     * @return The RMSE of the predictions.
     */
    public static double rmse(MetricTarget<Regressor> target, RegressionSufficientStatistics sufficientStats) {
        return compute(target, sufficientStats, RegressionMetrics::rmse);
    }

    /**
     * Calculates the RMSE based on the supplied statistics for a single dimension.
     * @param variable The regression dimension to target.
     * @param sufficientStats The sufficient statistics.
     * @return The RMSE of the predictions.
     */
    public static double rmse(Regressor variable, RegressionSufficientStatistics sufficientStats) {
        String varname = variable.getNames()[0];
        double sumSqErr = sufficientStats.sumSquaredError.get(varname).doubleValue();
        return Math.sqrt(sumSqErr / sufficientStats.weightSum);
    }

    /**
     * Calculates the Mean Absolute Error based on the supplied statistics.
     * @param target The regression dimension or average to target.
     * @param sufficientStats The sufficient statistics.
     * @return The MAE of the predictions.
     */
    public static double mae(MetricTarget<Regressor> target, RegressionSufficientStatistics sufficientStats) {
        return compute(target, sufficientStats, RegressionMetrics::mae);
    }

    /**
     * Calculates the Mean Absolute Error based on the supplied statistics for a single dimension.
     * @param variable The regression dimension to target.
     * @param sufficientStats The sufficient statistics.
     * @return The MAE of the predictions.
     */
    public static double mae(Regressor variable, RegressionSufficientStatistics sufficientStats) {
        String varname = variable.getNames()[0];
        double sumAbsErr = sufficientStats.sumAbsoluteError.get(varname).doubleValue();
        return sumAbsErr / sufficientStats.weightSum;
    }

    /**
     * Calculates the explained variance based on the supplied statistics.
     * @param target The regression dimension or average to target.
     * @param sufficientStats The sufficient statistics.
     * @return The explained variance of the truth given the predictions.
     */
    public static double explainedVariance(MetricTarget<Regressor> target, RegressionSufficientStatistics sufficientStats) {
        return compute(target, sufficientStats, RegressionMetrics::explainedVariance);
    }

    /**
     * Calculates the explained variance based on the supplied statistics for a single dimension.
     * @param variable The regression dimension to target.
     * @param sufficientStats The sufficient statistics.
     * @return The explained variance of the truth given the predictions.
     */
    public static double explainedVariance(Regressor variable, RegressionSufficientStatistics sufficientStats) {
        String varname = variable.getNames()[0];
        double[] trueArray = sufficientStats.trueValues.get(varname);
        double[] predictedArray = sufficientStats.predictedValues.get(varname);

        double meanDifference = 0.0;
        for (int i = 0; i < sufficientStats.n; i++) {
            meanDifference += sufficientStats.weights[i] * (trueArray[i] - predictedArray[i]);
        }
        meanDifference /= sufficientStats.weightSum;
        double meanTruth = Util.weightedMean(trueArray, sufficientStats.weights, sufficientStats.n);

        double numerator = 0d;
        double denominator = 0d;
        for (int i = 0; i < sufficientStats.n; i++) {
            float weight = sufficientStats.weights[i];
            double variance = trueArray[i] - predictedArray[i] - meanDifference;
            numerator += weight * variance * variance;
            double difference = trueArray[i] - meanTruth;
            denominator += weight * difference * difference;
        }

        return 1d - (numerator/denominator);
    }

    /**
     * Computes the supplied function on the supplied metric target.
     * @param target The metric target.
     * @param sufficientStats The sufficient statistics.
     * @param impl The function to apply.
     * @return The metric value.
     */
    private static double compute(MetricTarget<Regressor> target,
                                  RegressionSufficientStatistics sufficientStats,
                                  BiFunction<Regressor, RegressionSufficientStatistics, Double> impl) {
        if (target.getOutputTarget().isPresent()) {
            return impl.apply(target.getOutputTarget().get(), sufficientStats);
        } else if (target.getAverageTarget().isPresent()) {
            EvaluationMetric.Average averageType = target.getAverageTarget().get();
            switch (averageType) {
                case MACRO:
                    double accumulator = 0.0;
                    for (Regressor r : sufficientStats.domain.getDomain()) {
                        accumulator += impl.apply(r,sufficientStats);
                    }
                    return accumulator / sufficientStats.domain.size();
                case MICRO:
                    throw new IllegalStateException("Micro averages are not supported for regression metrics.");
                default:
                    throw new IllegalStateException("Unexpected average type " + averageType);
            }
        } else {
            throw new IllegalStateException("MetricTarget without target.");
        }
    }

}