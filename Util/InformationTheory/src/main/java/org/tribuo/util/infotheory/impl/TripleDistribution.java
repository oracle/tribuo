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

import java.util.HashMap;
import java.util.LinkedHashMap;
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
public class TripleDistribution<T1,T2,T3> {
    public static final int DEFAULT_MAP_SIZE = 20;

    public final long count;

    private final Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount;
    private final Map<CachedPair<T1,T2>,MutableLong> abCount;
    private final Map<CachedPair<T1,T3>,MutableLong> acCount;
    private final Map<CachedPair<T2,T3>,MutableLong> bcCount;
    private final Map<T1,MutableLong> aCount;
    private final Map<T2,MutableLong> bCount;
    private final Map<T3,MutableLong> cCount;

    public TripleDistribution(long count, Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount, Map<CachedPair<T1,T2>,MutableLong> abCount, Map<CachedPair<T1,T3>,MutableLong> acCount, Map<CachedPair<T2,T3>,MutableLong> bcCount, Map<T1,MutableLong> aCount, Map<T2,MutableLong> bCount, Map<T3,MutableLong> cCount) {
        this.count = count;
        this.jointCount = jointCount;
        this.abCount = abCount;
        this.acCount = acCount;
        this.bcCount = bcCount;
        this.aCount = aCount;
        this.bCount = bCount;
        this.cCount = cCount;
    }

    public Map<CachedTriple<T1,T2,T3>,MutableLong> getJointCount() {
        return jointCount;
    }
    
    public Map<CachedPair<T1,T2>,MutableLong> getABCount() {
        return abCount;
    }
    
    public Map<CachedPair<T1,T3>,MutableLong> getACCount() {
        return acCount;
    }
    
    public Map<CachedPair<T2,T3>,MutableLong> getBCCount() {
        return bcCount;
    }
    
    public Map<T1,MutableLong> getACount() {
        return aCount;
    }
    
    public Map<T2,MutableLong> getBCount() {
        return bCount;
    }
    
    public Map<T3,MutableLong> getCCount() {
        return cCount;
    }
    
    public static <T1,T2,T3> TripleDistribution<T1,T2,T3> constructFromLists(List<T1> first, List<T2> second, List<T3> third) {
        Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount = new LinkedHashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T1,T2>,MutableLong> abCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T1,T3>,MutableLong> acCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T2,T3>,MutableLong> bcCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T1,MutableLong> aCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T2,MutableLong> bCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T3,MutableLong> cCount = new HashMap<>(DEFAULT_MAP_SIZE);

        long count = first.size();

        if ((first.size() == second.size()) && (first.size() == third.size())) {
            for (int i = 0; i < first.size(); i++) {
                T1 a = first.get(i);
                T2 b = second.get(i);
                T3 c = third.get(i);
                CachedTriple<T1,T2,T3> abc = new CachedTriple<>(a,b,c);
                CachedPair<T1,T2> ab = abc.getAB();
                CachedPair<T1,T3> ac = abc.getAC();
                CachedPair<T2,T3> bc = abc.getBC();

                MutableLong abcCurCount = jointCount.computeIfAbsent(abc, k -> new MutableLong());
                abcCurCount.increment();

                MutableLong abCurCount = abCount.computeIfAbsent(ab, k -> new MutableLong());
                abCurCount.increment();
                MutableLong acCurCount = acCount.computeIfAbsent(ac, k -> new MutableLong());
                acCurCount.increment();
                MutableLong bcCurCount = bcCount.computeIfAbsent(bc, k -> new MutableLong());
                bcCurCount.increment();

                MutableLong aCurCount = aCount.computeIfAbsent(a, k -> new MutableLong());
                aCurCount.increment();
                MutableLong bCurCount = bCount.computeIfAbsent(b, k -> new MutableLong());
                bCurCount.increment();
                MutableLong cCurCount = cCount.computeIfAbsent(c, k -> new MutableLong());
                cCurCount.increment();
            }

            return new TripleDistribution<>(count,jointCount,abCount,acCount,bcCount,aCount,bCount,cCount);
        } else {
            throw new IllegalArgumentException("Counting requires lists of the same length. first.size() = " + first.size() + ", second.size() = " + second.size() + ", third.size() = " + third.size());
        }
    }

    public static <T1,T2,T3> TripleDistribution<T1,T2,T3> constructFromMap(Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount) {
        Map<CachedPair<T1,T2>,MutableLong> abCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T1,T3>,MutableLong> acCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T2,T3>,MutableLong> bcCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T1,MutableLong> aCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T2,MutableLong> bCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T3,MutableLong> cCount = new HashMap<>(DEFAULT_MAP_SIZE);

        return constructFromMap(jointCount,abCount,acCount,bcCount,aCount,bCount,cCount);
    }

    public static <T1,T2,T3> TripleDistribution<T1,T2,T3> constructFromMap(Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount,
                                                                           int abSize, int acSize, int bcSize,
                                                                           int aSize, int bSize, int cSize) {
        Map<CachedPair<T1,T2>,MutableLong> abCount = new HashMap<>(abSize);
        Map<CachedPair<T1,T3>,MutableLong> acCount = new HashMap<>(acSize);
        Map<CachedPair<T2,T3>,MutableLong> bcCount = new HashMap<>(bcSize);
        Map<T1,MutableLong> aCount = new HashMap<>(aSize);
        Map<T2,MutableLong> bCount = new HashMap<>(bSize);
        Map<T3,MutableLong> cCount = new HashMap<>(cSize);

        return constructFromMap(jointCount,abCount,acCount,bcCount,aCount,bCount,cCount);
    }

    public static <T1,T2,T3> TripleDistribution<T1,T2,T3> constructFromMap(Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount,
                                                                           Map<CachedPair<T1,T2>,MutableLong> abCount,
                                                                           Map<CachedPair<T1,T3>,MutableLong> acCount,
                                                                           Map<CachedPair<T2,T3>,MutableLong> bcCount,
                                                                           Map<T1,MutableLong> aCount,
                                                                           Map<T2,MutableLong> bCount,
                                                                           Map<T3,MutableLong> cCount) {
        long count = 0L;

        for (Entry<CachedTriple<T1,T2,T3>,MutableLong> e : jointCount.entrySet()) {
            CachedTriple<T1,T2,T3> abc = e.getKey();
            long curCount = e.getValue().longValue();
            CachedPair<T1,T2> ab = abc.getAB();
            CachedPair<T1,T3> ac = abc.getAC();
            CachedPair<T2,T3> bc = abc.getBC();
            T1 a = abc.getA();
            T2 b = abc.getB();
            T3 c = abc.getC();

            count += curCount;

            MutableLong abCurCount = abCount.computeIfAbsent(ab, k -> new MutableLong());
            abCurCount.increment(curCount);
            MutableLong acCurCount = acCount.computeIfAbsent(ac, k -> new MutableLong());
            acCurCount.increment(curCount);
            MutableLong bcCurCount = bcCount.computeIfAbsent(bc, k -> new MutableLong());
            bcCurCount.increment(curCount);

            MutableLong aCurCount = aCount.computeIfAbsent(a, k -> new MutableLong());
            aCurCount.increment(curCount);
            MutableLong bCurCount = bCount.computeIfAbsent(b, k -> new MutableLong());
            bCurCount.increment(curCount);
            MutableLong cCurCount = cCount.computeIfAbsent(c, k -> new MutableLong());
            cCurCount.increment(curCount);
        }

        return new TripleDistribution<>(count,jointCount,abCount,acCount,bcCount,aCount,bCount,cCount);
    }
    
}
