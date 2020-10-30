/*
 * Copyright (c) 2020, Oracle and/or its affiliates. All rights reserved.
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
package org.tribuo.util;

import java.io.Serializable;
import java.util.Objects;

/**
 * An accumulator for online calculation of the mean and variance of a
 * stream of doubles.
 * <p>
 * Note this class is not thread safe.
 */
public final class MeanVarianceAccumulator implements Serializable {
    private static final long serialVersionUID = 1L;

    private double max = Double.NEGATIVE_INFINITY;
    private double min = Double.POSITIVE_INFINITY;

    private double mean = 0.0;
    private double sumSquares = 0.0;

    private long count = 0;

    /**
     * Constructs an empty mean/variance accumulator.
     */
    public MeanVarianceAccumulator() {}

    /**
     * Constructs a mean/variance accumulator and observes the supplied array.
     * @param values The array to operate on.
     */
    public MeanVarianceAccumulator(double[] values) {
        observe(values);
    }

    /**
     * Copy constructor.
     * @param other The MeanVarianceAccumulator to copy.
     */
    public MeanVarianceAccumulator(MeanVarianceAccumulator other) {
        this.max = other.max;
        this.min = other.min;
        this.mean = other.mean;
        this.sumSquares = other.sumSquares;
        this.count = other.count;
    }

    /**
     * Resets this accumulator to the starting state.
     */
    public void reset() {
        this.max = Double.NEGATIVE_INFINITY;
        this.min = Double.POSITIVE_INFINITY;
        this.mean = 0;
        this.sumSquares = 0;
        this.count = 0;
    }

    /**
     * Observes a value, i.e. updates the mean, variance, max and min statistics.
     * @param value The value to observe.
     */
    public void observe(double value) {
        if (value < min) {
            min = value;
        }
        if (value > max) {
            max = value;
        }
        count++;
        double delta = value - mean;
        mean += delta / count;
        double delta2 = value - mean;
        sumSquares += delta * delta2;
    }

    /**
     * Observes an array of values, i.e. updates the mean, variance, max and min statistics.
     * @param values The values to observe.
     */
    public void observe(double[] values) {
        for (int i = 0; i < values.length; i++) {
            observe(values[i]);
        }
    }

    /**
     * Gets the minimum observed value.
     * @return The minimum value.
     */
    public double getMin() {
        return min;
    }

    /**
     * Gets the maximum observed value.
     * @return The maximum value.
     */
    public double getMax() {
        return max;
    }

    /**
     * Gets the sample mean.
     * @return The sample mean.
     */
    public double getMean() {
        return mean;
    }

    /**
     * Gets the sample variance.
     * @return The sample variance.
     */
    public double getVariance() {
        return sumSquares / (count-1);
    }

    /**
     * Gets the observation count.
     * @return The observation count.
     */
    public long getCount() {
        return count;
    }

    /**
     * Gets the sample standard deviation.
     * @return The sample standard deviation.
     */
    public double getStdDev() {
        return Math.sqrt(getVariance());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MeanVarianceAccumulator that = (MeanVarianceAccumulator) o;
        return Double.compare(that.max, max) == 0 &&
                Double.compare(that.min, min) == 0 &&
                Double.compare(that.mean, mean) == 0 &&
                Double.compare(that.sumSquares, sumSquares) == 0 &&
                count == that.count;
    }

    @Override
    public int hashCode() {
        return Objects.hash(max, min, mean, sumSquares, count);
    }

    @Override
    public String toString() {
        return String.format("Variable(count=%d,max=%f,min=%f,mean=%f,variance=%f)",count,max,min,mean,getVariance());
    }
}
