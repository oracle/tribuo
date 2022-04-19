/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.MutableNumber;
import org.tribuo.protos.core.CategoricalInfoProto;
import org.tribuo.protos.core.VariableInfoProto;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.stream.Collectors;

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
    private Object object;

    private static final long serialVersionUID = 2L;

    private static final MutableLong ZERO = new MutableLong(0);
    /**
     * The default threshold for converting a categorical info into a {@link RealInfo}.
     */
    public static final int THRESHOLD = 50;
    private static final double COMPARISON_THRESHOLD = 1e-10;

    /**
     * The occurrence counts of each value.
     */
    @ProtobufField(fieldName="key")
    @ProtobufField(fieldName="value")
    protected Map<Double,MutableLong> valueCounts = null;

    /**
     * The observed value if it's only seen a single one.
     */
    @ProtobufField(fieldName="key")
    protected double observedValue = Double.NaN;

    /**
     * The count of the observed value if it's only seen a single one.
     */
    @ProtobufField(fieldName="value")
    protected long observedCount = 0;

    // These variables are used in the sampling methods, and regenerated after serialization if a sample is required.
    /**
     * The values array.
     */
    protected transient double[] values = null;
    /**
     * The total number of observations (including zeros).
     */
    protected transient long totalObservations = -1;
    /**
     * The CDF to sample from.
     */
    protected transient double[] cdf = null;

    /**
     * Constructs a new empty categorical info for the supplied feature name.
     * @param name The feature name.
     */
    public CategoricalInfo(String name) {
        super(name);
    }

    /**
     * Constructs a deep copy of the supplied categorical info.
     * @param info The info to copy.
     */
    protected CategoricalInfo(CategoricalInfo info) {
        this(info,info.name);
    }

    /**
     * Constructs a deep copy of the supplied categorical info, with the new feature name.
     * @param info The info to copy.
     * @param newName The new feature name.
     */
    protected CategoricalInfo(CategoricalInfo info, String newName) {
        super(newName,info.count);
        if (info.valueCounts != null) {
            valueCounts = MutableNumber.copyMap(info.valueCounts);
        } else {
            observedValue = info.observedValue;
            observedCount = info.observedCount;
        }
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     */
    public static CategoricalInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        CategoricalInfoProto proto = message.unpack(CategoricalInfoProto.class);
        if (proto.getId() != -1) {
            throw new IllegalStateException("Invalid protobuf, found an id where none was expected, id = " + proto.getId());
        }
        CategoricalInfo info = new CategoricalInfo(proto.getName());
        List<Double> keys = proto.getKeyList();
        List<Long> values = proto.getValueList();
        if (keys.size() != values.size()) {
            throw new IllegalStateException("Invalid protobuf, keys and values don't match. keys.size() = " + keys.size() + ", values.size() = " + values.size());
        }
        int newCount = 0;
        if (keys.size() > 1) {
            info.valueCounts = new HashMap<>(keys.size());
            for (int i = 0; i < keys.size(); i++) {
                info.valueCounts.put(keys.get(i),new MutableLong(values.get(i)));
                newCount += values.get(i).intValue();
            }
        } else {
            info.observedValue = keys.get(0);
            info.observedCount = values.get(0);
            newCount = values.get(0).intValue();
        }
        info.count = newCount;
        return info;
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
            List<Map.Entry<Double, MutableLong>> entries = valueCounts.entrySet().stream()
                .sorted(Comparator.comparingDouble(Map.Entry::getKey)).collect(Collectors.toList());
            for (Map.Entry<Double, MutableLong> e : entries) {
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

            for (Map.Entry<Double, MutableLong> e : entries) {
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
            List<Map.Entry<Double, MutableLong>> entries = valueCounts.entrySet().stream()
                .sorted(Comparator.comparingDouble(Map.Entry::getKey)).collect(Collectors.toList());
            for (Map.Entry<Double,MutableLong> e : entries){
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
            for (Double key : valueCounts.keySet().stream().sorted().collect(Collectors.toList())) {
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
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        if (!super.equals(o)) {
            return false;
        }
        CategoricalInfo that = (CategoricalInfo) o;
        // MutableLong in OLCUT 5.2.0 doesn't implement equals,
        // so we can't compare valueCounts with Objects.equals.
        // That'll be fixed in the next OLCUT but for the time being we've got this workaround.
        if (valueCounts != null ^ that.valueCounts != null) {
            return false;
        } else if (valueCounts != null && that.valueCounts != null) {
            if (valueCounts.size() != that.valueCounts.size()) {
                return false;
            } else {
                for (Map.Entry<Double, MutableLong> e : valueCounts.entrySet()) {
                    MutableLong other = that.valueCounts.get(e.getKey());
                    if ((other == null) || (e.getValue().longValue() != other.longValue())) {
                        return false;
                    }
                }
            }
        }
        return Double.compare(that.observedValue, observedValue) == 0 && observedCount == that.observedCount;
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), valueCounts, observedValue, observedCount);
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

    @Override
    public VariableInfoProto serialize() {
        VariableInfoProto.Builder builder = VariableInfoProto.newBuilder();

        CategoricalInfoProto.Builder categoricalBuilder = CategoricalInfoProto.newBuilder();
        categoricalBuilder.setName(name);
        categoricalBuilder.setCount(count);
        categoricalBuilder.setId(-1);
        if (valueCounts != null) {
            for (Map.Entry<Double, MutableLong> e : valueCounts.entrySet()) {
                categoricalBuilder.addKey(e.getKey());
                categoricalBuilder.addValue(e.getValue().longValue());
            }
        } else {
            categoricalBuilder.addKey(observedValue);
            categoricalBuilder.addValue(observedCount);
        }

        builder.setVersion(0);
        builder.setClassName(this.getClass().getName());
        builder.setSerializedData(Any.pack(categoricalBuilder.build()));

        return builder.build();
    }
}
