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

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.util.infotheory.InformationTheory;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * A count distribution over {@link CachedPair} objects.
 * @param <T1> The type of the first element
 * @param <T2> The type of the second element
 */
public class PairDistribution<T1,T2> {

    /**
     * The number of samples this distribution has seen.
     */
    public final long count;

    /**
     * The joint distribution.
     */
    public final Map<CachedPair<T1,T2>,MutableLong> jointCounts;
    /**
     * The first marginal distribution.
     */
    public final Map<T1,MutableLong> firstCount;
    /**
     * The second marginal distribution.
     */
    public final Map<T2,MutableLong> secondCount;

    /**
     * Constructs a pair distribution.
     * @param count The total sample count.
     * @param jointCounts The joint counts.
     * @param firstCount The first variable count.
     * @param secondCount The second variable count.
     */
    public PairDistribution(long count, Map<CachedPair<T1,T2>,MutableLong> jointCounts, Map<T1,MutableLong> firstCount, Map<T2,MutableLong> secondCount) {
        this.count = count;
        this.jointCounts = new LinkedHashMap<>(jointCounts);
        this.firstCount = new LinkedHashMap<>(firstCount);
        this.secondCount = new LinkedHashMap<>(secondCount);
    }

    /**
     * Constructs a pair distribution.
     * @param count The total sample count.
     * @param jointCounts The joint counts.
     * @param firstCount The first variable count.
     * @param secondCount The second variable count.
     */
    public PairDistribution(long count, LinkedHashMap<CachedPair<T1,T2>,MutableLong> jointCounts, LinkedHashMap<T1,MutableLong> firstCount, LinkedHashMap<T2,MutableLong> secondCount) {
        this.count = count;
        this.jointCounts = jointCounts;
        this.firstCount = firstCount;
        this.secondCount = secondCount;
    }
    
    /**
     * Generates the counts for two vectors. Returns a PairDistribution containing the joint
     * count, and the two marginal counts.
     * @param <T1> Type of the first array.
     * @param <T2> Type of the second array.
     * @param first An array of values.
     * @param second Another array of values.
     * @return The joint counts and the two marginal counts.
     */
    public static <T1,T2> PairDistribution<T1,T2> constructFromLists(List<T1> first, List<T2> second) {
        LinkedHashMap<CachedPair<T1,T2>,MutableLong> abCountDist = new LinkedHashMap<>(InformationTheory.DEFAULT_MAP_SIZE);
        LinkedHashMap<T1,MutableLong> aCountDist = new LinkedHashMap<>(InformationTheory.DEFAULT_MAP_SIZE);
        LinkedHashMap<T2,MutableLong> bCountDist = new LinkedHashMap<>(InformationTheory.DEFAULT_MAP_SIZE);

        if (first.size() == second.size()) {
            long count = 0;
            for (int i = 0; i < first.size(); i++) {
                T1 a = first.get(i);
                T2 b = second.get(i);
                CachedPair<T1,T2> pair = new CachedPair<>(a,b);

                MutableLong abCount = abCountDist.computeIfAbsent(pair, k -> new MutableLong());
                abCount.increment();

                MutableLong aCount = aCountDist.computeIfAbsent(a, k -> new MutableLong());
                aCount.increment();

                MutableLong bCount = bCountDist.computeIfAbsent(b, k -> new MutableLong());
                bCount.increment();

                count++;
            }

            return new PairDistribution<>(count,abCountDist,aCountDist,bCountDist);
        } else {
            throw new IllegalArgumentException("Counting requires arrays of the same length. first.size() = " + first.size() + ", second.size() = " + second.size());
        }
    }

    /**
     * Constructs a distribution from a joint count.
     * @param jointCount The joint count.
     * @param <T1> The type of the first variable.
     * @param <T2> The type of the second variable.
     * @return A pair distribution.
     */
    public static <T1,T2> PairDistribution<T1,T2> constructFromMap(Map<CachedPair<T1,T2>,MutableLong> jointCount) {
        Map<T1,MutableLong> aCount = new HashMap<>(InformationTheory.DEFAULT_MAP_SIZE);
        Map<T2,MutableLong> bCount = new HashMap<>(InformationTheory.DEFAULT_MAP_SIZE);

        return constructFromMap(jointCount,aCount,bCount);
    }

    /**
     * Constructs a distribution from a joint count.
     * @param jointCount The joint count.
     * @param aSize The initial size of the first marginal hash map.
     * @param bSize The initial size of the second marginal hash map.
     * @param <T1> The type of the first variable.
     * @param <T2> The type of the second variable.
     * @return A pair distribution.
     */
    public static <T1,T2> PairDistribution<T1,T2> constructFromMap(Map<CachedPair<T1,T2>,MutableLong> jointCount, int aSize, int bSize) {
        Map<T1,MutableLong> aCount = new HashMap<>(aSize);
        Map<T2,MutableLong> bCount = new HashMap<>(bSize);

        return constructFromMap(jointCount,aCount,bCount);
    }

    /**
     * Constructs a joint distribution from the counts.
     * @param jointCount The joint count.
     * @param aCount The first marginal count.
     * @param bCount The second marginal count.
     * @param <T1> The type of the first variable.
     * @param <T2> The type of the second variable.
     * @return A pair distribution.
     */
    public static <T1,T2> PairDistribution<T1,T2> constructFromMap(Map<CachedPair<T1,T2>,MutableLong> jointCount,
                                                                           Map<T1,MutableLong> aCount,
                                                                           Map<T2,MutableLong> bCount) {
        long count = 0L;
        
        for (Entry<CachedPair<T1,T2>,MutableLong> e : jointCount.entrySet()) {
            CachedPair<T1,T2> pair = e.getKey();
            long curCount = e.getValue().longValue();
            T1 a = pair.getA();
            T2 b = pair.getB();

            MutableLong curACount = aCount.computeIfAbsent(a, k -> new MutableLong());
            curACount.increment(curCount);

            MutableLong curBCount = bCount.computeIfAbsent(b, k -> new MutableLong());
            curBCount.increment(curCount);
            count += curCount;
        }

        return new PairDistribution<>(count,jointCount,aCount,bCount);
    }

}
