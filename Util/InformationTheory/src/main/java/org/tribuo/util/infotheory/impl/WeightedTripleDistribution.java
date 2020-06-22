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

import org.tribuo.util.infotheory.WeightedInformationTheory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Generates the counts for a triplet of vectors. Contains the joint
 * count, the three pairwise counts, and the three marginal counts.
 * @param <T1> Type of the first list.
 * @param <T2> Type of the second list.
 * @param <T3> Type of the third list.
 */
public class WeightedTripleDistribution<T1,T2,T3> {
    public static final int DEFAULT_MAP_SIZE = 20;

    public final long count;

    private final Map<CachedTriple<T1,T2,T3>,WeightCountTuple> jointCount;
    private final Map<CachedPair<T1,T2>,WeightCountTuple> abCount;
    private final Map<CachedPair<T1,T3>,WeightCountTuple> acCount;
    private final Map<CachedPair<T2,T3>,WeightCountTuple> bcCount;
    private final Map<T1,WeightCountTuple> aCount;
    private final Map<T2,WeightCountTuple> bCount;
    private final Map<T3,WeightCountTuple> cCount;

    public WeightedTripleDistribution(long count, Map<CachedTriple<T1,T2,T3>,WeightCountTuple> jointCount, Map<CachedPair<T1,T2>,WeightCountTuple> abCount, Map<CachedPair<T1,T3>,WeightCountTuple> acCount, Map<CachedPair<T2,T3>,WeightCountTuple> bcCount, Map<T1,WeightCountTuple> aCount, Map<T2,WeightCountTuple> bCount, Map<T3,WeightCountTuple> cCount) {
        this.count = count;
        this.jointCount = jointCount;
        this.abCount = abCount;
        this.acCount = acCount;
        this.bcCount = bcCount;
        this.aCount = aCount;
        this.bCount = bCount;
        this.cCount = cCount;
    }

    public Map<CachedTriple<T1,T2,T3>,WeightCountTuple> getJointCount() {
        return jointCount;
    }
    
    public Map<CachedPair<T1,T2>,WeightCountTuple> getABCount() {
        return abCount;
    }
    
    public Map<CachedPair<T1,T3>,WeightCountTuple> getACCount() {
        return acCount;
    }
    
    public Map<CachedPair<T2,T3>,WeightCountTuple> getBCCount() {
        return bcCount;
    }
    
    public Map<T1,WeightCountTuple> getACount() {
        return aCount;
    }
    
    public Map<T2,WeightCountTuple> getBCount() {
        return bCount;
    }
    
    public Map<T3,WeightCountTuple> getCCount() {
        return cCount;
    }
    
    public static <T1,T2,T3> WeightedTripleDistribution<T1,T2,T3> constructFromLists(List<T1> first, List<T2> second, List<T3> third, List<Double> weights) {
        Map<CachedTriple<T1,T2,T3>,WeightCountTuple> jointCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T1,T2>,WeightCountTuple> abCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T1,T3>,WeightCountTuple> acCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T2,T3>,WeightCountTuple> bcCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T1,WeightCountTuple> aCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T2,WeightCountTuple> bCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T3,WeightCountTuple> cCount = new HashMap<>(DEFAULT_MAP_SIZE);

        long count = first.size();

        if ((first.size() == second.size()) && (first.size() == third.size()) && (first.size() == weights.size())) {
            for (int i = 0; i < first.size(); i++) {
                double weight = weights.get(i);
                T1 a = first.get(i);
                T2 b = second.get(i);
                T3 c = third.get(i);
                CachedTriple<T1,T2,T3> triple = new CachedTriple<>(a,b,c);
                CachedPair<T1,T2> abPair = triple.getAB();
                CachedPair<T1,T3> acPair = triple.getAC();
                CachedPair<T2,T3> bcPair = triple.getBC();

                WeightCountTuple abcCurCount = jointCount.computeIfAbsent(triple,(k) -> new WeightCountTuple());
                abcCurCount.weight += weight;
                abcCurCount.count++;

                WeightCountTuple abCurCount = abCount.computeIfAbsent(abPair,(k) -> new WeightCountTuple());
                abCurCount.weight += weight;
                abCurCount.count++;

                WeightCountTuple acCurCount = acCount.computeIfAbsent(acPair,(k) -> new WeightCountTuple());
                acCurCount.weight += weight;
                acCurCount.count++;
                
                WeightCountTuple bcCurCount = bcCount.computeIfAbsent(bcPair,(k) -> new WeightCountTuple());
                bcCurCount.weight += weight;
                bcCurCount.count++;
                
                WeightCountTuple aCurCount = aCount.computeIfAbsent(a,(k) -> new WeightCountTuple());
                aCurCount.weight += weight;
                aCurCount.count++;

                WeightCountTuple bCurCount = bCount.computeIfAbsent(b,(k) -> new WeightCountTuple());
                bCurCount.weight += weight;
                bCurCount.count++;

                WeightCountTuple cCurCount = cCount.computeIfAbsent(c,(k) -> new WeightCountTuple());
                cCurCount.weight += weight;
                cCurCount.count++;
            }

            WeightedInformationTheory.normaliseWeights(jointCount);
            WeightedInformationTheory.normaliseWeights(abCount);
            WeightedInformationTheory.normaliseWeights(acCount);
            WeightedInformationTheory.normaliseWeights(bcCount);
            WeightedInformationTheory.normaliseWeights(aCount);
            WeightedInformationTheory.normaliseWeights(bCount);
            WeightedInformationTheory.normaliseWeights(cCount);

            return new WeightedTripleDistribution<>(count,jointCount,abCount,acCount,bcCount,aCount,bCount,cCount);
        } else {
            throw new IllegalArgumentException("Counting requires lists of the same length. first.size() = " + first.size() + ", second.size() = " + second.size() + ", third.size() = " + third.size() + ", weights.size() = " + weights.size());
        }
    }

    public static <T1,T2,T3> WeightedTripleDistribution<T1,T2,T3> constructFromMap(Map<CachedTriple<T1,T2,T3>,WeightCountTuple> jointCount) {
        Map<CachedPair<T1,T2>,WeightCountTuple> abCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T1,T3>,WeightCountTuple> acCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T2,T3>,WeightCountTuple> bcCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T1,WeightCountTuple> aCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T2,WeightCountTuple> bCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T3,WeightCountTuple> cCount = new HashMap<>(DEFAULT_MAP_SIZE);
        
        long count = 0L;

        for (Entry<CachedTriple<T1,T2,T3>,WeightCountTuple> e : jointCount.entrySet()) {
            CachedTriple<T1,T2,T3> triple = e.getKey();
            WeightCountTuple tuple = e.getValue();
            CachedPair<T1,T2> abPair = triple.getAB();
            CachedPair<T1,T3> acPair = triple.getAC();
            CachedPair<T2,T3> bcPair = triple.getBC();
            T1 a = triple.getA();
            T2 b = triple.getB();
            T3 c = triple.getC();

            count += tuple.count;

            double weight = tuple.weight * tuple.count;

            WeightCountTuple abCurCount = abCount.computeIfAbsent(abPair,(k) -> new WeightCountTuple());
            abCurCount.weight += weight;
            abCurCount.count += tuple.count;

            WeightCountTuple acCurCount = acCount.computeIfAbsent(acPair,(k) -> new WeightCountTuple());
            acCurCount.weight += weight;
            acCurCount.count += tuple.count;

            WeightCountTuple bcCurCount = bcCount.computeIfAbsent(bcPair,(k) -> new WeightCountTuple());
            bcCurCount.weight += weight;
            bcCurCount.count += tuple.count;

            WeightCountTuple aCurCount = aCount.computeIfAbsent(a,(k) -> new WeightCountTuple());
            aCurCount.weight += weight;
            aCurCount.count += tuple.count;

            WeightCountTuple bCurCount = bCount.computeIfAbsent(b,(k) -> new WeightCountTuple());
            bCurCount.weight += weight;
            bCurCount.count += tuple.count;

            WeightCountTuple cCurCount = cCount.computeIfAbsent(c,(k) -> new WeightCountTuple());
            cCurCount.weight += weight;
            cCurCount.count += tuple.count;
        }

        WeightedInformationTheory.normaliseWeights(abCount);
        WeightedInformationTheory.normaliseWeights(acCount);
        WeightedInformationTheory.normaliseWeights(bcCount);
        WeightedInformationTheory.normaliseWeights(aCount);
        WeightedInformationTheory.normaliseWeights(bCount);
        WeightedInformationTheory.normaliseWeights(cCount);

        return new WeightedTripleDistribution<>(count,jointCount,abCount,acCount,bcCount,aCount,bCount,cCount);
    }
    
}
