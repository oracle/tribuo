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

package org.tribuo.math.optimisers.util;

import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.la.VectorIterator;
import org.tribuo.math.la.VectorTuple;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/**
 * A subclass of {@link DenseVector} which shrinks the value every time a new value is added.
 * <p>
 * Be careful when modifying this or {@link DenseVector}.
 */
public class ShrinkingVector extends DenseVector implements ShrinkingTensor {
    private final double baseRate;
    private final boolean scaleShrinking;
    private final double lambdaSqrt;
    private final boolean reproject;
    private double squaredTwoNorm;
    private int iteration;
    private double multiplier;

    /**
     * Constructs a shrinking vector copy of the supplied dense matrix.
     * <p>
     * This vector shrinks during each call to {@link #intersectAndAddInPlace(Tensor, DoubleUnaryOperator)}.
     * @param v The vector to copy.
     * @param baseRate The base amount of shrinking to apply after each update.
     * @param scaleShrinking If true reduce the shrinking value over time proportionally to the number of updates.
     */
    public ShrinkingVector(DenseVector v, double baseRate, boolean scaleShrinking) {
        super(v);
        this.baseRate = baseRate;
        this.scaleShrinking = scaleShrinking;
        this.lambdaSqrt = 0.0;
        this.reproject = false;
        this.iteration = 1;
        this.multiplier = 1.0;
    }

    /**
     * Constructs a shrinking vector copy of the supplied dense vector.
     * <p>
     * This vector shrinks during each call to {@link #intersectAndAddInPlace(Tensor, DoubleUnaryOperator)},
     * and then reprojects the vector so it has the same twoNorm.
     * @param v The vector to copy.
     * @param baseRate The base rate of shrinkage.
     * @param lambda The lambda value (see {@link org.tribuo.math.optimisers.Pegasos}).
     */
    public ShrinkingVector(DenseVector v, double baseRate, double lambda) {
        super(v);
        this.baseRate = baseRate;
        this.scaleShrinking = true;
        this.lambdaSqrt = Math.sqrt(lambda);
        this.reproject = true;
        this.squaredTwoNorm = 0.0;
        this.iteration = 1;
        this.multiplier = 1.0;
    }

    private ShrinkingVector(double[] values, double baseRate, boolean scaleShrinking, double lambdaSqrt, boolean reproject, double squaredTwoNorm, int iteration, double multiplier) {
        super(values);
        this.baseRate = baseRate;
        this.scaleShrinking = scaleShrinking;
        this.lambdaSqrt = lambdaSqrt;
        this.reproject = reproject;
        this.squaredTwoNorm = squaredTwoNorm;
        this.iteration = iteration;
        this.multiplier = multiplier;
    }

    @Override
    public DenseVector convertToDense() {
        return DenseVector.createDenseVector(toArray());
    }

    @Override
    public ShrinkingVector copy() {
        return new ShrinkingVector(Arrays.copyOf(elements,elements.length),baseRate,scaleShrinking,lambdaSqrt,reproject,squaredTwoNorm,iteration,multiplier);
    }

    @Override
    public double[] toArray() {
        double[] newValues = new double[elements.length];
        for (int i = 0; i < newValues.length; i++) {
            newValues[i] = get(i);
        }
        return newValues;
    }

    @Override
    public double get(int index) {
        return elements[index] * multiplier;
    }

    @Override
    public double sum() {
        double sum = 0.0;
        for (int i = 0; i < elements.length; i++) {
            sum += get(i);
        }
        return sum;
    }

    @Override
    public void intersectAndAddInPlace(Tensor other, DoubleUnaryOperator f) {
        double shrinkage = scaleShrinking ? 1.0 - (baseRate / iteration) : 1.0 - baseRate;
        scaleInPlace(shrinkage);
        SGDVector otherVec = (SGDVector) other;
        for (VectorTuple tuple : otherVec) {
            double update = f.applyAsDouble(tuple.value);
            double oldValue = elements[tuple.index] * multiplier;
            double newValue = oldValue + update;
            squaredTwoNorm -= oldValue * oldValue;
            squaredTwoNorm += newValue * newValue;
            elements[tuple.index] = newValue / multiplier;
        }
        if (reproject) {
            double projectionNormaliser = (1.0 / lambdaSqrt) / twoNorm();
            if (projectionNormaliser < 1.0) {
                scaleInPlace(projectionNormaliser);
            }
        }
        iteration++;
    }

    @Override
    public int indexOfMax() {
        int index = 0;
        double value = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < elements.length; i++) {
            double tmp = get(i);
            if (tmp > value) {
                index = i;
                value = tmp;
            }
        }
        return index;
    }

    @Override
    public double dot(SGDVector other) {
        double score = 0.0;

        for (VectorTuple tuple : other) {
            score += get(tuple.index) * tuple.value;
        }

        return score;
    }

    @Override
    public void scaleInPlace(double value) {
        multiplier *= value;
        if (Math.abs(multiplier) < tolerance) {
            reifyMultiplier();
        }
    }

    private void reifyMultiplier() {
        for (int i = 0; i < elements.length; i++) {
            elements[i] *= multiplier;
        }
        multiplier = 1.0;
    }

    @Override
    public double twoNorm() {
        return Math.sqrt(squaredTwoNorm);
    }

    @Override
    public double maxValue() {
        return multiplier * super.maxValue();
    }

    @Override
    public double minValue() {
        return multiplier * super.minValue();
    }

    @Override
    public VectorIterator iterator() {
        return new ShrinkingVectorIterator(this);
    }

    private static class ShrinkingVectorIterator implements VectorIterator {
        private final ShrinkingVector vector;
        private final VectorTuple tuple;
        private int index;

        public ShrinkingVectorIterator(ShrinkingVector vector) {
            this.vector = vector;
            this.tuple = new VectorTuple();
            this.index = 0;
        }

        @Override
        public boolean hasNext() {
            return index < vector.size();
        }

        @Override
        public VectorTuple next() {
            tuple.index = index;
            tuple.value = vector.get(index);
            index++;
            return tuple;
        }

        @Override
        public VectorTuple getReference() {
            return tuple;
        }
    }
}


