/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.clustering.ClusterID;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.util.infotheory.InformationTheory;

import java.util.List;
import java.util.function.BiFunction;

/**
 * Default metrics for evaluating clusterings.
 */
public enum ClusteringMetrics {

    /**
     * The normalized mutual information between the two clusterings
     */
    NORMALIZED_MI((target, context) -> ClusteringMetrics.normalizedMI(context)),
    /**
     * The normalized mutual information adjusted for chance.
     */
    ADJUSTED_MI((target, context) -> ClusteringMetrics.adjustedMI(context));

    private final BiFunction<MetricTarget<ClusterID>, ClusteringMetric.Context, Double> impl;

    ClusteringMetrics(BiFunction<MetricTarget<ClusterID>, ClusteringMetric.Context, Double> impl) {
        this.impl = impl;
    }

    /**
     * Gets the implementing function for this metric.
     * @return The implementing function.
     */
    public BiFunction<MetricTarget<ClusterID>, ClusteringMetric.Context, Double> getImpl() {
        return impl;
    }

    /**
     * Constructs the metric for the specified metric target.
     * @param tgt The metric target.
     * @return The clustering metric for that target.
     */
    public ClusteringMetric forTarget(MetricTarget<ClusterID> tgt) {
        return new ClusteringMetric(tgt, this.name(), this.getImpl());
    }

    /**
     * Calculates the adjusted normalized mutual information between two clusterings.
     * @param context The context containing the predicted clustering and the ground truth.
     * @return The adjusted normalized mutual information.
     */
    public static double adjustedMI(ClusteringMetric.Context context) {
        return adjustedMI(context.getPredictedIDs(), context.getTrueIDs());
    }

    /**
     * Calculates the adjusted normalized mutual information between two clusterings.
     * @param predictedIDs The predicted cluster ids.
     * @param trueIDs The ground truth cluster ids.
     * @return The adjusted normalized mutual information.
     */
    public static double adjustedMI(List<Integer> predictedIDs, List<Integer> trueIDs) {
        double mi = InformationTheory.mi(predictedIDs, trueIDs);
        double predEntropy = InformationTheory.entropy(predictedIDs);
        double trueEntropy = InformationTheory.entropy(trueIDs);
        double expectedMI = InformationTheory.expectedMI(trueIDs, predictedIDs);

        double minEntropy = Math.min(predEntropy, trueEntropy);
        double denominator = minEntropy - expectedMI;
        
        if (denominator < 0) {
            denominator = Math.min(denominator, -2.220446049250313e-16);
        } else {
            denominator = Math.max(denominator, 2.220446049250313e-16);
        }


        return (mi - expectedMI) / (denominator);
    }

    /**
     * Calculates the normalized mutual information between two clusterings.
     * @param context The context containing the predicted clustering and the ground truth.
     * @return The normalized mutual information.
     */
    public static double normalizedMI(ClusteringMetric.Context context) {
        double mi = InformationTheory.mi(context.getPredictedIDs(), context.getTrueIDs());
        double predEntropy = InformationTheory.entropy(context.getPredictedIDs());
        double trueEntropy = InformationTheory.entropy(context.getTrueIDs());

        return predEntropy < trueEntropy ? mi / predEntropy : mi / trueEntropy;
    }

}