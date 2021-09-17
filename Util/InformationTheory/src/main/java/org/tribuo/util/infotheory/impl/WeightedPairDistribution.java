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

package org.tribuo.util.infotheory.impl;

import org.tribuo.util.infotheory.InformationTheory;
import org.tribuo.util.infotheory.WeightedInformationTheory;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Generates the counts for a pair of vectors. Contains the joint
 * count and the two marginal counts.
 * @param <T1> Type of the first list.
 * @param <T2> Type of the second list.
 */
public class WeightedPairDistribution<T1,T2> {

    /**
     * The sample count.
     */
    public final long count;

    private final Map<CachedPair<T1,T2>,WeightCountTuple> jointCounts;
    private final Map<T1,WeightCountTuple> firstCount;
    private final Map<T2,WeightCountTuple> secondCount;

    /**
     * Constructs a weighted pair distribution from the supplied values.
     * <p>
     * Copies the maps out into LinkedHashMaps for iteration speed.
     * @param count The sample count.
     * @param jointCounts The joint distribution.
     * @param firstCount The first marginal distribution.
     * @param secondCount The second marginal distribution.
     */
    public WeightedPairDistribution(long count, Map<CachedPair<T1,T2>,WeightCountTuple> jointCounts, Map<T1,WeightCountTuple> firstCount, Map<T2,WeightCountTuple> secondCount) {
        this.count = count;
        this.jointCounts = new LinkedHashMap<>(jointCounts);
        this.firstCount = new LinkedHashMap<>(firstCount);
        this.secondCount = new LinkedHashMap<>(secondCount);
    }

    /**
     * Constructs a weighted pair distribution from the supplied values.
     * @param count The sample count.
     * @param jointCounts The joint distribution.
     * @param firstCount The first marginal distribution.
     * @param secondCount The second marginal distribution.
     */
    public WeightedPairDistribution(long count, LinkedHashMap<CachedPair<T1,T2>,WeightCountTuple> jointCounts, LinkedHashMap<T1,WeightCountTuple> firstCount, LinkedHashMap<T2,WeightCountTuple> secondCount) {
        this.count = count;
        this.jointCounts = jointCounts;
        this.firstCount = firstCount;
        this.secondCount = secondCount;
    }

    /**
     * Gets the joint distribution.
     * @return The joint distribution.
     */
    public Map<CachedPair<T1,T2>,WeightCountTuple> getJointCounts() {
        return jointCounts;
    }

    /**
     * Gets the first marginal distribution.
     * @return The first marginal distribution.
     */
    public Map<T1,WeightCountTuple> getFirstCount() {
        return firstCount;
    }

    /**
     * Gets the second marginal distribution.
     * @return The second marginal distribution.
     */
    public Map<T2,WeightCountTuple> getSecondCount() {
        return secondCount;
    }
    
    /**
     * Generates the counts for two vectors. Returns a pair containing the joint
     * count, and a pair of the two marginal counts.
     * @param <T1> Type of the first list.
     * @param <T2> Type of the second list.
     * @param first An list of values.
     * @param second Another list of values.
     * @param weights An list of per example weights.
     * @return A WeightedPairDistribution.
     */
    public static <T1,T2> WeightedPairDistribution<T1,T2> constructFromLists(List<T1> first, List<T2> second, List<Double> weights) {
        LinkedHashMap<CachedPair<T1,T2>,WeightCountTuple> countDist = new LinkedHashMap<>(InformationTheory.DEFAULT_MAP_SIZE);
        LinkedHashMap<T1,WeightCountTuple> aCountDist = new LinkedHashMap<>(InformationTheory.DEFAULT_MAP_SIZE);
        LinkedHashMap<T2,WeightCountTuple> bCountDist = new LinkedHashMap<>(InformationTheory.DEFAULT_MAP_SIZE);

        if ((first.size() == second.size()) && (first.size() == weights.size())) {
            long count = 0;
            for (int i = 0; i < first.size(); i++) {
                T1 a = first.get(i);
                T2 b = second.get(i);
                double weight = weights.get(i);
                CachedPair<T1,T2> pair = new CachedPair<>(a,b);

                WeightCountTuple abCurCount = countDist.computeIfAbsent(pair,(k) -> new WeightCountTuple());
                abCurCount.weight += weight;
                abCurCount.count++;

                WeightCountTuple aCurCount = aCountDist.computeIfAbsent(a,(k) -> new WeightCountTuple());
                aCurCount.weight += weight;
                aCurCount.count++;

                WeightCountTuple bCurCount = bCountDist.computeIfAbsent(b,(k) -> new WeightCountTuple());
                bCurCount.weight += weight;
                bCurCount.count++;

                count++;
            }

            WeightedInformationTheory.normaliseWeights(countDist);
            WeightedInformationTheory.normaliseWeights(aCountDist);
            WeightedInformationTheory.normaliseWeights(bCountDist);

            return new WeightedPairDistribution<>(count,countDist,aCountDist,bCountDist);
        } else {
            throw new IllegalArgumentException("Counting requires lists of the same length. first.size() = " + first.size() + ", second.size() = " + second.size() + ", weights.size() = " + weights.size());
        }
    }

    /**
     * Generates a WeightedPairDistribution by generating the marginal distributions for the first and second elements.
     * This assumes the weights have already been normalised.
     * @param <T1> Type of the first element.
     * @param <T2> Type of the second element.
     * @param jointCount The (normalised) input map.
     * @return A WeightedPairDistribution
     */
    public static <T1,T2> WeightedPairDistribution<T1,T2> constructFromMap(Map<CachedPair<T1,T2>,WeightCountTuple> jointCount) {
        LinkedHashMap<CachedPair<T1,T2>,WeightCountTuple> countDist = new LinkedHashMap<>(jointCount);
        LinkedHashMap<T1,WeightCountTuple> aCountDist = new LinkedHashMap<>(InformationTheory.DEFAULT_MAP_SIZE);
        LinkedHashMap<T2,WeightCountTuple> bCountDist = new LinkedHashMap<>(InformationTheory.DEFAULT_MAP_SIZE);

        long count = 0L;
        
        for (Entry<CachedPair<T1,T2>,WeightCountTuple> e : countDist.entrySet()) {
            CachedPair<T1,T2> pair = e.getKey();
            WeightCountTuple tuple = e.getValue();
            T1 a = pair.getA();
            T2 b = pair.getB();
            double weight = tuple.weight * tuple.count;

            WeightCountTuple aCurCount = aCountDist.computeIfAbsent(a,(k) -> new WeightCountTuple());
            aCurCount.weight += weight;
            aCurCount.count += tuple.count;

            WeightCountTuple bCurCount = bCountDist.computeIfAbsent(b,(k) -> new WeightCountTuple());
            bCurCount.weight += weight;
            bCurCount.count += tuple.count;

            count += tuple.count;
        }

        WeightedInformationTheory.normaliseWeights(aCountDist);
        WeightedInformationTheory.normaliseWeights(bCountDist);

        return new WeightedPairDistribution<>(count,countDist,aCountDist,bCountDist);
    }

}
