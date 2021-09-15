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

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.clustering.ClusterID;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.util.infotheory.InformationTheory;
import org.tribuo.util.infotheory.impl.PairDistribution;
import org.apache.commons.math3.special.Gamma;

import java.util.List;
import java.util.Map;
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
        double mi = InformationTheory.mi(context.getPredictedIDs(), context.getTrueIDs());
        double predEntropy = InformationTheory.entropy(context.getPredictedIDs());
        double trueEntropy = InformationTheory.entropy(context.getTrueIDs());
        double expectedMI = expectedMI(context.getPredictedIDs(), context.getTrueIDs());

        double minEntropy = Math.min(predEntropy, trueEntropy);

        return (mi - expectedMI) / (minEntropy - expectedMI);
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

    private static double expectedMI(List<Integer> first, List<Integer> second) {
        PairDistribution<Integer,Integer> pd = PairDistribution.constructFromLists(first,second);

        Map<Integer, MutableLong> firstCount = pd.firstCount;
        Map<Integer,MutableLong> secondCount = pd.secondCount;
        long count = pd.count;

        double output = 0.0;

        for (Map.Entry<Integer,MutableLong> f : firstCount.entrySet()) {
            for (Map.Entry<Integer,MutableLong> s : secondCount.entrySet()) {
                long fVal = f.getValue().longValue();
                long sVal = s.getValue().longValue();
                long minCount = Math.min(fVal, sVal);

                long threshold = fVal + sVal - count;
                long start = threshold > 1 ? threshold : 1;

                for (long nij = start; nij < minCount; nij++) {
                    double acc = ((double) nij) / count;
                    acc *= Math.log(((double) (count * nij)) / (fVal * sVal));
                    //numerator
                    double logSpace = Gamma.logGamma(fVal + 1);
                    logSpace += Gamma.logGamma(sVal + 1);
                    logSpace += Gamma.logGamma(count - fVal + 1);
                    logSpace += Gamma.logGamma(count - sVal + 1);
                    //denominator
                    logSpace -= Gamma.logGamma(count + 1);
                    logSpace -= Gamma.logGamma(nij + 1);
                    logSpace -= Gamma.logGamma(fVal - nij + 1);
                    logSpace -= Gamma.logGamma(sVal - nij + 1);
                    logSpace -= Gamma.logGamma(count - fVal - sVal + nij + 1);
                    acc *= Math.exp(logSpace);
                    output += acc;
                }
            }
        }
        return output;
    }

}