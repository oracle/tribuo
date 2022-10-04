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

package org.tribuo.regression;

import com.oracle.labs.mlrg.olcut.util.MutableDouble;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.MutableNumber;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputInfo;
import org.tribuo.regression.Regressor.DimensionTuple;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * The base class for regression information using {@link Regressor}s.
 * <p>
 * Stores the observed min, max, mean and variance for each dimension.
 */
public abstract class RegressionInfo implements OutputInfo<Regressor> {
    private static final long serialVersionUID = 2L;

    private static final MutableDouble NAN = new MutableDouble(Double.NaN);

    /**
     * The maximum values seen per dimension.
     */
    protected Map<String,MutableDouble> maxMap = new LinkedHashMap<>();
    /**
     * The minimum values seen per dimension.
     */
    protected Map<String,MutableDouble> minMap = new LinkedHashMap<>();

    /**
     * The mean value per dimension.
     */
    protected Map<String,MutableDouble> meanMap = new LinkedHashMap<>();
    /**
     * Sum of the squared differences per dimension.
     */
    protected Map<String,MutableDouble> sumSquaresMap = new LinkedHashMap<>();

    /**
     * The occurrence count per dimension.
     */
    protected Map<String,MutableLong> countMap = new TreeMap<>();

    /**
     * The overall number of {@link Regressor} objects this object has seen.
     */
    protected long overallCount = 0;

    /**
     * The number of unknown {@link Regressor}s this object has seen.
     */
    protected int unknownCount = 0;

    /**
     * Creates an empty RegressionInfo.
     */
    RegressionInfo() { }

    /**
     * Copies a RegressionInfo, apart from the unknownCount.
     * @param other The RegressionInfo to copy.
     */
    RegressionInfo(RegressionInfo other) {
        this.maxMap = MutableNumber.copyMap(other.maxMap);
        this.minMap = MutableNumber.copyMap(other.minMap);
        this.meanMap = MutableNumber.copyMap(other.meanMap);
        this.sumSquaresMap = MutableNumber.copyMap(other.sumSquaresMap);
        this.countMap = MutableNumber.copyMap(other.countMap);
        this.overallCount = other.overallCount;
    }

    RegressionInfo(Map<String,MutableLong> countMap, Map<String,MutableDouble> maxMap, Map<String,MutableDouble> minMap, Map<String,MutableDouble> meanMap, Map<String,MutableDouble> sumSquaresMap, int unknownCount, long overallCount) {
        this.countMap = MutableNumber.copyMap(countMap);
        for (Map.Entry<String,MutableLong> e : this.countMap.entrySet()) {
            if (e.getValue().longValue() < 1) {
                throw new IllegalArgumentException("Invalid count found for dimension name '" + e.getKey() + "', expected positive value found " + e.getValue());
            }
        }
        this.maxMap = MutableNumber.copyMap(maxMap);
        this.minMap = MutableNumber.copyMap(minMap);
        this.meanMap = MutableNumber.copyMap(meanMap);
        this.sumSquaresMap = MutableNumber.copyMap(sumSquaresMap);
        this.unknownCount = unknownCount;
        this.overallCount = overallCount;
    }

    @Override
    public int getUnknownCount() {
        return unknownCount;
    }

    /**
     * Returns a set containing a Regressor for each dimension with the minimum value observed.
     * @return A set of Regressors, each with one active dimension.
     */
    @Override
    public Set<Regressor> getDomain() {
        TreeSet<DimensionTuple> outputs = new TreeSet<>(Comparator.comparing(DimensionTuple::getName));
        for (Map.Entry<String,MutableDouble> e : minMap.entrySet()) {
            outputs.add(new DimensionTuple(e.getKey(),e.getValue().doubleValue()));
        }
        @SuppressWarnings("unchecked") // DimensionTuple is a subtype of Regressor, and this set is immutable.
        SortedSet<Regressor> setOutputs = (SortedSet<Regressor>) (SortedSet) Collections.unmodifiableSortedSet(outputs);
        return setOutputs;
    }

    /**
     * Gets the minimum value this RegressionInfo has seen, or NaN if it's not seen anything.
     * @param name The dimension to check.
     * @return The minimum value for that dimension.
     */
    public double getMin(String name) {
        return minMap.getOrDefault(name,NAN).doubleValue();
    }

    /**
     * Gets the maximum value this RegressionInfo has seen, or NaN if it's not seen that dimension.
     * @param name The dimension to check.
     * @return The maximum value for that dimension.
     */
    public double getMax(String name) {
        return maxMap.getOrDefault(name,NAN).doubleValue();
    }

    /**
     * Gets the mean value this RegressionInfo has seen, or NaN if it's not seen that dimension.
     * @param name The dimension to check.
     * @return The mean value for that dimension.
     */
    public double getMean(String name) {
        return meanMap.getOrDefault(name,NAN).doubleValue();
    }

    /**
     * Gets the variance this RegressionInfo has seen, or NaN if it's not seen that dimension.
     * @param name The dimension to check.
     * @return The variance for that dimension.
     */
    public double getVariance(String name) {
        MutableDouble sumSquaresDbl = sumSquaresMap.get(name);
        if (sumSquaresDbl != null) {
            return sumSquaresDbl.doubleValue() / (countMap.get(name).longValue()-1);
        } else {
            return Double.NaN;
        }
    }

    /**
     * The number of dimensions this OutputInfo has seen.
     * @return The number of dimensions this OutputInfo has seen.
     */
    @Override
    public int size() {
        return countMap.size();
    }

    @Override
    public ImmutableOutputInfo<Regressor> generateImmutableOutputInfo() {
        return new ImmutableRegressionInfo(this);
    }

    @Override
    public MutableOutputInfo<Regressor> generateMutableOutputInfo() {
        return new MutableRegressionInfo(this);
    }

    @Override
    public abstract RegressionInfo copy();

    @Override
    public Iterable<Pair<String,Long>> outputCountsIterable() {
        ArrayList<Pair<String,Long>> list = new ArrayList<>();

        for (Map.Entry<String,MutableLong> e : countMap.entrySet()) {
            list.add(new Pair<>(e.getKey(), e.getValue().longValue()));
        }

        return list;
    }

    static boolean checkMutableDouble(MutableDouble boxA, MutableDouble boxB) {
        double a = boxA.doubleValue();
        double b = boxB.doubleValue();
        return Double.compare(a,b) == 0;
    }
}
