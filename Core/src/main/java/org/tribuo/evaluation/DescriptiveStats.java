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

package org.tribuo.evaluation;

import org.tribuo.util.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Descriptive statistics calculated across a list of doubles.
 */
public final class DescriptiveStats {

    private final List<Double> samples = new ArrayList<>();

    /**
     * Create an empty DescriptiveStats.
     */
    public DescriptiveStats() {}

    /**
     * Create a DescriptiveStats initialized with the supplied values.
     * @param values The initial values.
     */
    public DescriptiveStats(List<Double> values) {
        this.samples.addAll(values);
    }

    /**
     * Package private method for appending a value to a DescriptiveStats.
     * @param value A value to append.
     */
    void addValue(double value) {
        samples.add(value);
    }

    /**
     * Calculates the mean of the values.
     * @return The mean.
     */
    public double getMean() {
        return Util.mean(samples);
    }

    /**
     * Calculates the sample variance of the values.
     * @return The sample variance.
     */
    public double getVariance() {
        return Util.sampleVariance(samples);
    }

    /**
     * Calculates the standard deviation of the values.
     * @return The standard deviation.
     */
    public double getStandardDeviation() {
        return Util.sampleStandardDeviation(samples);
    }

    /**
     * Calculates the max of the values.
     * @return The maximum value.
     */
    public double getMax() {
        return Util.argmax(samples).getB();
    }

    /**
     * Calculates the min of the values.
     * @return The minimum value.
     */
    public double getMin() {
        return Util.argmin(samples).getB();
    }

    /**
     * Returns the number of values.
     * @return The number of values.
     */
    public long getN() {
        return samples.size();
    }

    /**
     * Returns a copy of the values.
     * @return A copy of the values.
     */
    public List<Double> values() {
        return new ArrayList<>(samples);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DescriptiveStats that = (DescriptiveStats) o;
        return samples.equals(that.samples);
    }

    @Override
    public int hashCode() {
        return Objects.hash(samples);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        List<String> rows = Arrays.asList("count", "mean", "std", "min", "max");
        int maxRowLen = rows.stream().max(Comparator.comparingInt(String::length)).get().length();
        String fmtStr = String.format("%%-%ds", maxRowLen+2);

        sb.append(String.format(fmtStr, "count"));
        sb.append(String.format("%d%n", getN()));

        sb.append(String.format(fmtStr, "mean"));
        sb.append(String.format("%.6f%n", getMean()));

        sb.append(String.format(fmtStr, "std"));
        sb.append(String.format("%.6f%n", getStandardDeviation()));

        sb.append(String.format(fmtStr, "min"));
        sb.append(String.format("%.6f%n", getMin()));

        sb.append(String.format(fmtStr, "max"));
        sb.append(String.format("%.6f%n", getMax()));

        return sb.toString();
    }
}