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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.MutableNumber;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.SplittableRandom;

/**
 * Stores information about Categorical features.
 * <p>
 * Contains a mapping from values to observed counts for that value, has
 * an initial optimisation for the binary case to reduce memory consumption.
 * </p>
 * <p>
 * Can be transformed into a {@link RealInfo} if there are too many unique observed values.
 * </p>
 * <p>
 * Does not contain an id number, but can be transformed into {@link CategoricalIDInfo} which
 * does contain an id number.
 * </p>
 * <p>
 * Note that the synchronization in this class only protects instantiation where CDF and values
 * are recomputed. Care should be taken if data is read while {@link #observe(double)} is called.
 * </p>
 */
public class CategoricalInfo extends SkeletalVariableInfo {
    private static final long serialVersionUID = 2L;

    private static final MutableLong ZERO = new MutableLong(0);
    /**
     * The default threshold for converting a categorical info into a {@link RealInfo}.
     */
    public static final int THRESHOLD = 50;
    private static final double COMPARISON_THRESHOLD = 1e-10;

    protected Map<Double,MutableLong> valueCounts = null;

    protected double observedValue = Double.NaN;

    protected long observedCount = 0;

    // These variables are used in the sampling methods, and regenerated after serialization if a sample is required.
    protected transient double[] values = null;
    protected transient long totalObservations = -1;
    protected transient double[] cdf = null;

    public CategoricalInfo(String name) {
        super(name);
    }

    protected CategoricalInfo(CategoricalInfo info) {
        this(info,info.name);
    }

    protected CategoricalInfo(CategoricalInfo info, String newName) {
        super(newName,info.count);
        if (info.valueCounts != null) {
            valueCounts = MutableNumber.copyMap(info.valueCounts);
        } else {
            observedValue = info.observedValue;
            observedCount = info.observedCount;
        }
    }

    @Override
    protected void observe(double value) {
        if (value != 0.0) {
            super.observe(value);
            if (valueCounts != null) {
                MutableLong count = valueCounts.computeIfAbsent(value, k -> new MutableLong());
                count.increment();
            } else {
                if (Double.isNaN(observedValue)) {
                    observedValue = value;
                    observedCount++;
                } else if (Math.abs(value - observedValue) < COMPARISON_THRESHOLD) {
                    observedCount++;
                } else {
                    // Observed two values for this CategoricalInfo, now it needs a HashMap.
                    valueCounts = new HashMap<>(4);
                    valueCounts.put(observedValue, new MutableLong(observedCount));
                    valueCounts.put(value, new MutableLong(1));
                    observedValue = Double.NaN;
                    observedCount = 0;
                }
            }
            values = null;
        }
    }

    /**
     * Gets the number of times a specific value was observed, and zero if this value is unknown.
     * @param value The value to check.
     * @return The count of times this value was observed, zero otherwise.
     */
    public long getObservationCount(double value) {
        if (valueCounts != null) {
            return valueCounts.getOrDefault(value, ZERO).longValue();
        } else {
            if (Math.abs(value - observedValue) < COMPARISON_THRESHOLD) {
                return observedCount;
            } else {
                return 0;
            }
        }
    }

    /**
     * Gets the number of unique values this CategoricalInfo has observed.
     * @return An int representing the number of unique values.
     */
    public int getUniqueObservations() {
        if (valueCounts != null) {
            return valueCounts.size();
        } else {
            if (Double.isNaN(observedValue)) {
                return 0;
            } else {
                return 1;
            }
        }
    }

    /**
     * Generates a {@link RealInfo} using the currently observed counts to calculate
     * the min, max, mean and variance.
     * @return A RealInfo representing the data in this CategoricalInfo.
     */
    public RealInfo generateRealInfo() {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double sum = 0.0;
        double sumSquares = 0.0;
        double mean;

        if (valueCounts != null) {
            for (Map.Entry<Double, MutableLong> e : valueCounts.entrySet()) {
                double value = e.getKey();
                double valCount = e.getValue().longValue();
                if (value > max) {
                    max = value;
                }
                if (value < min) {
                    min = value;
                }
                sum += value * valCount;
            }
            mean = sum / count;

            for (Map.Entry<Double, MutableLong> e : valueCounts.entrySet()) {
                double value = e.getKey();
                double valCount = e.getValue().longValue();
                sumSquares += (value - mean) * (value - mean) * valCount;
            }
        } else {
            min = observedValue;
            max = observedValue;
            mean = observedValue;
            sumSquares = 0.0;
        }

        return new RealInfo(name,count,max,min,mean,sumSquares);
    }

    @Override
    public CategoricalInfo copy() {
        return new CategoricalInfo(this);
    }

    @Override
    public CategoricalIDInfo makeIDInfo(int id) {
        return new CategoricalIDInfo(this,id);
    }

    @Override
    public CategoricalInfo rename(String newName) {
        return new CategoricalInfo(this,newName);
    }

    @Override
    public synchronized double uniformSample(SplittableRandom rng) {
        if (values == null) {
            regenerateValues();
        }
        int sampleIdx = rng.nextInt(values.length);
        return values[sampleIdx];
    }

    /**
     * Samples a value from this feature according to the frequency of observation.
     * @param rng The RNG to use.
     * @param totalObservations The observations including the implicit zeros.
     * @return The sampled value.
     */
    public double frequencyBasedSample(SplittableRandom rng, long totalObservations) {
        if ((totalObservations != this.totalObservations) || (cdf == null)) {
            regenerateCDF(totalObservations);
        }
        int lookup = Util.sampleFromCDF(cdf,rng);
        return values[lookup];
    }

    /**
     * Samples a value from this feature according to the frequency of observation.
     * @param rng The RNG to use.
     * @param totalObservations The observations including the implicit zeros.
     * @return The sampled value.
     */
    public double frequencyBasedSample(Random rng, long totalObservations) {
        if ((totalObservations != this.totalObservations) || (cdf == null)) {
            regenerateCDF(totalObservations);
        }
        int lookup = Util.sampleFromCDF(cdf,rng);
        return values[lookup];
    }

    /**
     * Generates the CDF for sampling.
     * @param newTotalObservations The new number of total observations including the implicit zeros.
     */
    private synchronized void regenerateCDF(long newTotalObservations) {
        long[] counts;
        if (valueCounts != null) {
            // This is tricksy as if valueCounts contains zero that means
            // we could have both observed zeros and unobserved zeros.
            if (valueCounts.containsKey(0.0)) {
                values = new double[valueCounts.size()];
                counts = new long[valueCounts.size()];
            } else {
                values = new double[valueCounts.size()+1];
                counts = new long[valueCounts.size()+1];
            }
            values[0] = 0;
            counts[0] = newTotalObservations;
            int counter = 1;
            long total = 0;
            for (Map.Entry<Double,MutableLong> e : valueCounts.entrySet()) {
                if (e.getKey() != 0.0) {
                    values[counter] = e.getKey();
                    counts[counter] = e.getValue().longValue();
                    total += counts[counter];
                    counter++;
                }
            }
            // Set the zero counts appropriately
            counts[0] -= total;
        } else {
            if (Double.isNaN(observedValue) || observedValue == 0.0) {
                values = new double[1];
                counts = new long[1];
                values[0] = 0;
                counts[0] = newTotalObservations;
            } else {
                values = new double[2];
                counts = new long[2];
                values[0] = 0;
                counts[0] = newTotalObservations - observedCount;
                values[1] = observedValue;
                counts[1] = observedCount;
            }
        }
        long sum = 0;
        for (int i = 0; i < counts.length; i++) {
            sum += counts[i];
        }
        if (sum != newTotalObservations) {
            throw new IllegalStateException("Total counts = " + sum + ", supplied value = " + newTotalObservations);
        }
        cdf = Util.generateCDF(counts,sum);
        totalObservations = newTotalObservations;
    }

    /**
     * Recomputes the values array.
     */
    private synchronized void regenerateValues() {
        //
        // Recompute values array
        if (valueCounts != null) {
            int counter;
            if (valueCounts.containsKey(0.0)) {
                values = new double[valueCounts.size()];
                counter = 0;
            } else {
                values = new double[valueCounts.size() + 1];
                values[0] = 0;
                counter = 1;
            }
            for (Double key : valueCounts.keySet()) {
                values[counter] = key;
                counter++;
            }
        } else {
            if (Double.isNaN(observedValue) || observedValue == 0.0) {
                values = new double[1];
                values[0] = 0;
            } else {
                values = new double[2];
                values[0] = 0;
                values[1] = observedValue;
            }
        }
    }

    @Override
    public String toString() {
        if (valueCounts != null) {
            return "CategoricalFeature(name=" + name + ",count=" + count + ",map=" + valueCounts.toString() + ")";
        } else {
            return "CategoricalFeature(name=" + name + ",count=" + count + ",map={" +observedValue+","+observedCount+"})";
        }
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        totalObservations = -1;
        values = null;
        cdf = null;
    }
}
