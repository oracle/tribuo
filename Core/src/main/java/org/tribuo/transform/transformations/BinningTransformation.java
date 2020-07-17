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

package org.tribuo.transform.transformations;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.EnumProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance;
import org.tribuo.transform.TransformStatistics;
import org.tribuo.transform.Transformation;
import org.tribuo.transform.TransformationProvenance;
import org.tribuo.transform.Transformer;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A Transformation which bins values.
 * <p>
 * Three binning types are implemented:
 * <ul>
 * <li>Equal width bins, based on the observed min and max.</li>
 * <li>Equal frequency bins, based on the observed data.   </li>
 * <li>Standard deviation width bins, based on the observed standard deviation and mean.</li>
 * </ul>
 * <p>
 * The equal frequency {@link TransformStatistics} needs to
 * store all the observed feature values, and thus has much higher
 * memory usage than all other binning types.
 * <p>
 * The binned values are in the range [1, numBins].
 */
public final class BinningTransformation implements Transformation {

    /**
     * The allowed binning types.
     */
    public enum BinningType { EQUAL_WIDTH, EQUAL_FREQUENCY, STD_DEVS }

    private static final String NUM_BINS = "numBins";
    private static final String TYPE = "type";

    @Config(description="Number of bins.")
    private int numBins;

    @Config(description="Binning algorithm to use.")
    private BinningType type;

    /**
     * For olcut.
     */
    private BinningTransformation() { }

    private BinningTransformation(BinningType type, int numBins) {
        this.type = type;
        this.numBins = numBins;
        postConfig();
    }

    @Override
    public void postConfig() {
        if (numBins < 2) {
            throw new IllegalArgumentException("Number of bins must be 2 or greater, found " + numBins);
        } else if (type == BinningType.STD_DEVS && ((numBins & 1) == 1)) {
            throw new IllegalArgumentException("Std dev must have an even number of bins, found " + numBins);
        }
    }

    @Override
    public TransformStatistics createStats() {
        switch (type) {
            case EQUAL_WIDTH:
                return new EqualWidthStats(numBins);
            case EQUAL_FREQUENCY:
                return new EqualFreqStats(numBins);
            case STD_DEVS:
                return new StdDevStats(numBins);
            default:
                throw new IllegalStateException("Unknown binning type " + type);
        }
    }

    @Override
    public TransformationProvenance getProvenance() {
        return new BinningTransformationProvenance(this);
    }

    public final static class BinningTransformationProvenance implements TransformationProvenance {
        private static final long serialVersionUID = 1L;

        private final IntProvenance numBins;
        private final EnumProvenance<BinningType> type;

        BinningTransformationProvenance(BinningTransformation host) {
            this.numBins = new IntProvenance(NUM_BINS,host.numBins);
            this.type = new EnumProvenance<>(TYPE,host.type);
        }

        @SuppressWarnings("unchecked") // Enum cast
        public BinningTransformationProvenance(Map<String,Provenance> map) {
            numBins = ObjectProvenance.checkAndExtractProvenance(map,NUM_BINS,IntProvenance.class,BinningTransformationProvenance.class.getSimpleName());
            type = ObjectProvenance.checkAndExtractProvenance(map,TYPE,EnumProvenance.class,BinningTransformationProvenance.class.getSimpleName());
        }

        @Override
        public String getClassName() {
            return BinningTransformation.class.getName();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof BinningTransformationProvenance)) return false;
            BinningTransformationProvenance pairs = (BinningTransformationProvenance) o;
            return numBins.equals(pairs.numBins) &&
                    type.equals(pairs.type);
        }

        @Override
        public int hashCode() {
            return Objects.hash(numBins, type);
        }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            Map<String,Provenance> map = new HashMap<>();
            map.put(NUM_BINS,numBins);
            map.put(TYPE,type);
            return Collections.unmodifiableMap(map);
        }
    }

    @Override
    public String toString() {
        return "BinningTransformation(type="+type+",numBins="+numBins+")";
    }

    /**
     * Returns a BinningTransformation which generates
     * fixed equal width bins between the observed min and max
     * values.
     * <p>
     * Values outside the observed range are clamped to either
     * the minimum or maximum bin. Bins are numbered in the range
     * [1,numBins].
     * @param numBins The number of bins to generate.
     * @return An equal width binning.
     */
    public static BinningTransformation equalWidth(int numBins) {
        return new BinningTransformation(BinningType.EQUAL_WIDTH,numBins);
    }

    /**
     * Returns a BinningTransformation which generates
     * bins which contain the same amount of training data
     * that is, each bin has an equal probability of occurrence
     * in the training data.
     * <p>
     * Values outside the observed range are clamped to either
     * the minimum or maximum bin. Bins are numbered in the range
     * [1,numBins].
     * @param numBins The number of bins to generate.
     * @return An equal frequency binning.
     */
    public static BinningTransformation equalFrequency(int numBins) {
        return new BinningTransformation(BinningType.EQUAL_FREQUENCY,numBins);
    }

    /**
     * Returns a BinningTransformation which generates bins
     * based on the observed standard deviation of the training
     * data. Each bin is a standard deviation wide, except for
     * the bins at the edges which encompass all lower or higher
     * values.
     * <p>
     * Bins are numbered in the range [1,numDeviations*2]. The middle two
     * bins are either side of the mean, the lowest bin is the mean minus
     * numDeviations * observed standard deviation, the highest bin is the
     * mean plus numDeviations * observed standard deviation.
     * @param numDeviations The number of standard deviations to bin.
     * @return A standard deviation based binning.
     */
    public static BinningTransformation stdDevs(int numDeviations) {
        return new BinningTransformation(BinningType.STD_DEVS,numDeviations*2);
    }

    private static class EqualWidthStats implements TransformStatistics {
        private final int numBins;

        private double min = Double.POSITIVE_INFINITY;
        private double max = Double.NEGATIVE_INFINITY;

        public EqualWidthStats(int numBins) {
            this.numBins = numBins;
        }

        @Override
        public void observeValue(double value) {
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }

        @Override
        public void observeSparse() { }

        @Override
        public void observeSparse(int count) { }

        @Override
        public Transformer generateTransformer() {
            double range = Math.abs(max - min);
            double increment = range / numBins;
            double[] bins = new double[numBins];
            double[] values = new double[numBins];

            for (int i = 0; i < bins.length; i++) {
                bins[i] = min + ((i+1) * increment);
                values[i] = i+1;
            }

            return new BinningTransformer(BinningType.EQUAL_WIDTH,bins,values);
        }

        @Override
        public String toString() {
            return "EqualWidthStatistics(min="+min+",max="+max+",numBins="+numBins+")";
        }
    }

    private static class EqualFreqStats implements TransformStatistics {
        private static final int DEFAULT_SIZE = 50;
        private final int numBins;

        private double[] observedValues;
        private int count;

        public EqualFreqStats(int numBins) {
            this.numBins = numBins;
            this.observedValues = new double[DEFAULT_SIZE];
            this.count = 0;
        }

        @Override
        public void observeValue(double value) {
            if (observedValues.length == count + 1) {
                growArray();
            }
            observedValues[count] = value;
            count++;
        }

        protected void growArray(int minCapacity) {
            int newCapacity = newCapacity(minCapacity);
            observedValues = Arrays.copyOf(observedValues,newCapacity);
        }

        /**
         * Returns a capacity at least as large as the given minimum capacity.
         * Returns the current capacity increased by 50% if that suffices.
         * Will not return a capacity greater than MAX_ARRAY_SIZE unless
         * the given minimum capacity is greater than MAX_ARRAY_SIZE.
         *
         * @param minCapacity the desired minimum capacity
         * @throws OutOfMemoryError if minCapacity is less than zero
         */
        private int newCapacity(int minCapacity) {
            // overflow-conscious code
            int oldCapacity = observedValues.length;
            int newCapacity = oldCapacity + (oldCapacity >> 1);
            if (newCapacity - minCapacity <= 0) {
                if (minCapacity < 0) // overflow
                    throw new OutOfMemoryError();
                return minCapacity;
            }
            return newCapacity;
        }

        protected void growArray() {
            growArray(count+1);
        }

        @Override
        public void observeSparse() { }

        @Override
        public void observeSparse(int count) { }

        @Override
        public Transformer generateTransformer() {
            if (numBins > observedValues.length) {
                throw new IllegalStateException("Needs more values than bins, requested " + numBins + " bins, but only found " + observedValues.length + " values.");
            }
            Arrays.sort(observedValues,0,count);
            double[] bins = new double[numBins];
            double[] values = new double[numBins];
            int increment = count / numBins;
            for (int i = 0; i < numBins-1; i++) {
                bins[i] = observedValues[(i+1)*increment];
                values[i] = i+1;
            }
            bins[numBins-1] = observedValues[count-1];
            values[numBins-1] = numBins;
            return new BinningTransformer(BinningType.EQUAL_FREQUENCY, bins, values);
        }

        @Override
        public String toString() {
            return "EqualFreqStatistics(count="+count+",numBins="+numBins+")";
        }
    }

    private static class StdDevStats implements TransformStatistics {
        private final int numBins;

        private double mean = 0;
        private double sumSquares = 0;
        private long count = 0;

        public StdDevStats(int numBins) {
            this.numBins = numBins;
        }

        @Override
        public void observeValue(double value) {
            count++;
            double delta = value - mean;
            mean += delta / count;
            double delta2 = value - mean;
            sumSquares += delta * delta2;
        }

        @Override
        public void observeSparse() { }

        @Override
        public void observeSparse(int count) { }

        @Override
        public Transformer generateTransformer() {
            double[] bins = new double[numBins];
            double[] values = new double[numBins];

            double stdDev = Math.sqrt(sumSquares/(count-1));

            int binCount = -(numBins/2);

            for (int i = 0; i < bins.length; i++) {
                values[i] = i+1;
                binCount++;
                bins[i] = mean + (binCount * stdDev);
            }

            return new BinningTransformer(BinningType.STD_DEVS,bins,values);
        }

        @Override
        public String toString() {
            return "StdDevStatistics(mean="+mean+",sumSquares="+sumSquares+",count="+count+",numBins="+numBins+")";
        }
    }

    private static class BinningTransformer implements Transformer {
        private static final long serialVersionUID = 1L;

        private final BinningType type;
        private final double[] bins;
        private final double[] values;

        public BinningTransformer(BinningType type, double[] bins, double[] values) {
            this.type = type;
            this.bins = bins;
            this.values = values;
        }

        @Override
        public double transform(double input) {
            if (input > bins[bins.length-1]) {
                return values[bins.length-1];
            } else {
                int index = Arrays.binarySearch(bins,input);
                if (index < 0) {
                    return values[- 1 - index];
                } else {
                    return values[index];
                }
            }
        }

        @Override
        public String toString() {
            return "BinningTransformer(type="+type+",bins="+Arrays.toString(bins)+",values="+Arrays.toString(values)+")";
        }
    }
}
