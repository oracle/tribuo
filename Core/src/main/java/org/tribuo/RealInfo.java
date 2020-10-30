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

import java.util.SplittableRandom;

/**
 * Stores information about real valued features.
 * <p>
 * Contains sufficient statistics to model the feature as a gaussian, plus the max and min values.
 * <p>
 * Does not contain an id number, but can be transformed into {@link RealIDInfo} which
 * does contain an id number.
 */
public class RealInfo extends SkeletalVariableInfo {
    private static final long serialVersionUID = 1L;

    /**
     * The maximum observed feature value.
     */
    protected double max = Double.NEGATIVE_INFINITY;

    /**
     * The minimum observed feature value.
     */
    protected double min = Double.POSITIVE_INFINITY;

    /**
     * The feature mean.
     */
    protected double mean = 0.0;

    /**
     * The sum of the squared feature values (used to compute the variance).
     */
    protected double sumSquares = 0.0;

    /**
     * Creates an empty real info with the supplied name.
     * @param name The feature name.
     */
    public RealInfo(String name) {
        super(name);
    }

    /**
     * Creates a real info with the supplied starting conditions.
     * <p>
     * All observations are assumed to be of zero.
     * @param name The feature name.
     * @param count The number of zeros observed.
     */
    public RealInfo(String name, int count) {
        super(name, count);
    }

    /**
     * Creates a real info with the supplied starting conditions.
     * @param name The feature name.
     * @param count The observation count.
     * @param max The maximum observed value.
     * @param min The minimum observed value.
     * @param mean The mean observed value.
     * @param sumSquares The sum of the squared values (used to calculate variance online).
     */
    public RealInfo(String name, int count, double max, double min, double mean, double sumSquares) {
        super(name, count);
        this.max = max;
        this.min = min;
        this.mean = mean;
        this.sumSquares = sumSquares;
    }

    /**
     * Copy constructor.
     * @param other The info to copy.
     */
    public RealInfo(RealInfo other) {
        this(other,other.name);
    }

    /**
     * Copy constructor which renames the feature. Used to redact the feature name.
     * @param other The info to copy.
     * @param newName The new name.
     */
    protected RealInfo(RealInfo other, String newName) {
        super(newName,other.count);
        this.max = other.max;
        this.min = other.min;
        this.mean = other.mean;
        this.sumSquares = other.sumSquares;
    }

    @Override
    protected void observe(double value) {
        if (value != 0.0) {
            super.observe(value);
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
            double delta = value - mean;
            mean += delta / count;
            double delta2 = value - mean;
            sumSquares += delta * delta2;
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

    @Override
    public RealInfo copy() {
        return new RealInfo(this);
    }

    @Override
    public RealIDInfo makeIDInfo(int id) {
        return new RealIDInfo(this,id);
    }

    @Override
    public RealInfo rename(String newName) {
        return new RealInfo(this,newName);
    }

    @Override
    public double uniformSample(SplittableRandom rng) {
        return (rng.nextDouble()*max) - min;
    }

    @Override
    public String toString() {
        return String.format("RealFeature(name=%s,count=%d,max=%f,min=%f,mean=%f,variance=%f)",name,count,max,min,mean,(sumSquares /(count-1)));
    }
}
