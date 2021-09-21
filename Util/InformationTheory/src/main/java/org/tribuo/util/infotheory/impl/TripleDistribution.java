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
 * <p>
 * Relies upon hashCode and equals to determine element equality for counting.
 * @param <T1> Type of the first list.
 * @param <T2> Type of the second list.
 * @param <T3> Type of the third list.
 */
public class TripleDistribution<T1,T2,T3> {
    /**
     * The default map size to initialise the marginalised count maps with.
     */
    public static final int DEFAULT_MAP_SIZE = 20;

    /**
     * The number of samples in this distribution.
     */
    public final long count;

    private final Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount;
    private final Map<CachedPair<T1,T2>,MutableLong> abCount;
    private final Map<CachedPair<T1,T3>,MutableLong> acCount;
    private final Map<CachedPair<T2,T3>,MutableLong> bcCount;
    private final Map<T1,MutableLong> aCount;
    private final Map<T2,MutableLong> bCount;
    private final Map<T3,MutableLong> cCount;

    /**
     * Constructs a triple distribution from the supplied distributions.
     * @param count The sample count.
     * @param jointCount The joint distribution over the three variables.
     * @param abCount The joint distribution over A and B.
     * @param acCount The joint distribution over A and C.
     * @param bcCount The joint distribution over B and C.
     * @param aCount The marginal distribution over A.
     * @param bCount The marginal distribution over B.
     * @param cCount The marginal distribution over C.
     */
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

    /**
     * The joint distribution over the three variables.
     * @return The joint distribution.
     */
    public Map<CachedTriple<T1,T2,T3>,MutableLong> getJointCount() {
        return jointCount;
    }

    /**
     * The joint distribution over the first and second variables.
     * @return The joint distribution over A and B.
     */
    public Map<CachedPair<T1,T2>,MutableLong> getABCount() {
        return abCount;
    }

    /**
     * The joint distribution over the first and third variables.
     * @return The joint distribution over A and C.
     */
    public Map<CachedPair<T1,T3>,MutableLong> getACCount() {
        return acCount;
    }

    /**
     * The joint distribution over the second and third variables.
     * @return The joint distribution over B and C.
     */
    public Map<CachedPair<T2,T3>,MutableLong> getBCCount() {
        return bcCount;
    }

    /**
     * The marginal distribution over the first variable.
     * @return The marginal distribution for A.
     */
    public Map<T1,MutableLong> getACount() {
        return aCount;
    }

    /**
     * The marginal distribution over the second variable.
     * @return The marginal distribution for B.
     */
    public Map<T2,MutableLong> getBCount() {
        return bCount;
    }

    /**
     * The marginal distribution over the third variable.
     * @return The marginal distribution for C.
     */
    public Map<T3,MutableLong> getCCount() {
        return cCount;
    }

    /**
     * Constructs a TripleDistribution from three lists of the same length.
     * <p>
     * If they are not the same length it throws IllegalArgumentException.
     * @param first The first list.
     * @param second The second list.
     * @param third The third list.
     * @param <T1> The first type.
     * @param <T2> The second type.
     * @param <T3> The third type.
     * @return The TripleDistribution.
     */
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

    /**
     * Constructs a TripleDistribution by marginalising the supplied joint distribution.
     * @param jointCount The joint distribution.
     * @param <T1> The type of A.
     * @param <T2> The type of B.
     * @param <T3> The type of C.
     * @return A TripleDistribution.
     */
    public static <T1,T2,T3> TripleDistribution<T1,T2,T3> constructFromMap(Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount) {
        Map<CachedPair<T1,T2>,MutableLong> abCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T1,T3>,MutableLong> acCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<CachedPair<T2,T3>,MutableLong> bcCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T1,MutableLong> aCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T2,MutableLong> bCount = new HashMap<>(DEFAULT_MAP_SIZE);
        Map<T3,MutableLong> cCount = new HashMap<>(DEFAULT_MAP_SIZE);

        return constructFromMap(jointCount,abCount,acCount,bcCount,aCount,bCount,cCount);
    }

    /**
     * Constructs a TripleDistribution by marginalising the supplied joint distribution.
     * <p>
     * Sizes are used to preallocate the HashMaps.
     * @param jointCount The joint distribution.
     * @param abSize The number of unique AB states.
     * @param acSize The number of unique AC states.
     * @param bcSize The number of unique BC states.
     * @param aSize The number of unique A states.
     * @param bSize The number of unique B states.
     * @param cSize The number of unique C states.
     * @param <T1> The type of A.
     * @param <T2> The type of B.
     * @param <T3> The type of C.
     * @return A TripleDistribution.
     */
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

    /**
     * Constructs a TripleDistribution by marginalising the supplied joint distribution.
     * <p>
     * Sizes are used to preallocate the HashMaps.
     * @param jointCount The joint distribution.
     * @param abCount An empty hashmap for AB.
     * @param acCount An empty hashmap for AC.
     * @param bcCount An empty hashmap for BC.
     * @param aCount An empty hashmap for A.
     * @param bCount An empty hashmap for B.
     * @param cCount An empty hashmap for C.
     * @param <T1> The type of A.
     * @param <T2> The type of B.
     * @param <T3> The type of C.
     * @return A TripleDistribution.
     */
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
