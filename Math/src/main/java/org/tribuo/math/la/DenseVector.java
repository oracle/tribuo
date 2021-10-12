/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.la;

import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.util.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleBiFunction;

/**
 * A dense vector, backed by a double array.
 */
public class DenseVector implements SGDVector {
    private static final long serialVersionUID = 1L;

    private final int[] shape;
    protected final double[] elements;

    /**
     * Creates an empty dense vector of the specified size.
     * @param size The vector size.
     */
    public DenseVector(int size) {
        this(size,0.0);
    }

    /**
     * Creates a dense vector of the specified size where each element is initialised to the specified value.
     * @param size The vector size.
     * @param value The initial value.
     */
    public DenseVector(int size, double value) {
        this.elements = new double[size];
        Arrays.fill(this.elements,value);
        this.shape = new int[]{size};
    }

    /**
     * Does not defensively copy the input, used internally.
     * @param values The values of this dense vector.
     */
    protected DenseVector(double[] values) {
        this.elements = values;
        this.shape = new int[]{elements.length};
    }

    /**
     * Copy constructor.
     * @param other The vector to copy.
     */
    protected DenseVector(DenseVector other) {
        this(other.toArray());
    }

    /**
     * Defensively copies the values before construction.
     * @param values The values of this dense vector.
     * @return A new dense vector.
     */
    public static DenseVector createDenseVector(double[] values) {
        return new DenseVector(Arrays.copyOf(values,values.length));
    }

    /**
     * Builds a {@link DenseVector} from an {@link Example}.
     * <p>
     * Used in training and inference.
     * <p>
     * Throws {@link IllegalArgumentException} if the Example contains NaN-valued features or
     * if no features in this Example are present in the feature map..
     * <p>
     * Unspecified features are set to zero.
     * @param example     The example to convert.
     * @param featureInfo The feature information, used to calculate the dimension of this DenseVector.
     * @param addBias     Add a bias feature.
     * @param <T>         The type parameter of the {@code example}.
     * @return A DenseVector representing the example's features.
     */
    public static <T extends Output<T>> DenseVector createDenseVector(Example<T> example, ImmutableFeatureMap featureInfo, boolean addBias) {
        int numFeatures = addBias ? featureInfo.size() + 1 : featureInfo.size();
        double[] values = new double[numFeatures];
        boolean found = false;
        for (Feature f : example) {
            int index = featureInfo.getID(f.getName());
            // If it's a valid feature for this feature map.
            if (index != -1) {
                values[index] = f.getValue();
                found = true;
                if (Double.isNaN(values[index])) {
                    throw new IllegalArgumentException("Example contained a NaN feature, " + f.toString());
                }
            }
        }
        if (!found) {
            throw new IllegalArgumentException("No features in this example were found in the feature map. Example - " + example.toString());
        }
        if (addBias) {
            values[numFeatures-1] = 1.0;
        }
        return new DenseVector(values);
    }

    /**
     * Generates a copy of the values in this DenseVector.
     * <p>
     * This implementation uses Arrays.copyOf, and should be overridden if the
     * get function has been modified.
     * @return A copy of the values in this DenseVector.
     */
    @Override
    public double[] toArray() {
        return Arrays.copyOf(elements, elements.length);
    }

    @Override
    public int[] getShape() {
        return shape;
    }

    @Override
    public Tensor reshape(int[] newShape) {
        int sum = Tensor.shapeSum(newShape);
        if (sum != elements.length) {
            throw new IllegalArgumentException("Invalid shape " + Arrays.toString(newShape) + ", expected something with " + elements.length + " elements.");
        }

        if (newShape.length == 2) {
            DenseMatrix matrix = new DenseMatrix(newShape[0],newShape[1]);

            for (int a = 0; a < size(); a++) {
                int i = a % newShape[0];
                int j = a / newShape[0];
                matrix.set(i,j,get(a));
            }

            return matrix;
        } else if (newShape.length == 1) {
            return new DenseVector(this);
        } else {
            throw new IllegalArgumentException("Only supports 1 or 2 dimensional tensors.");
        }
    }

    @Override
    public DenseVector copy() {
        return new DenseVector(toArray());
    }

    @Override
    public int size() {
        return elements.length;
    }

    @Override
    public int numActiveElements() {
        return elements.length;
    }

    /**
     * Performs a reduction from left to right of this vector.
     * @param initialValue The initial value.
     * @param op The element wise operation to apply before reducing.
     * @param reduction The reduction operation (should be commutative).
     * @return The reduced value.
     */
    @Override
    public double reduce(double initialValue, DoubleUnaryOperator op, DoubleBinaryOperator reduction) {
        double output = initialValue;
        for (int i = 0; i < elements.length; i++) {
            double transformed = op.applyAsDouble(get(i));
            output = reduction.applyAsDouble(transformed,output);
        }
        return output;
    }

    /**
     * Equals is defined mathematically, that is two SGDVectors are equal iff they have the same indices
     * and the same values at those indices.
     * @param other Object to compare against.
     * @return True if this vector and the other vector contain the same values in the same order.
     */
    @Override
    public boolean equals(Object other) {
        if (other instanceof SGDVector) {
            SGDVector otherVector = (SGDVector) other;
            if (elements.length == otherVector.size()) {
                Iterator<VectorTuple> ourItr = iterator();
                Iterator<VectorTuple> otherItr = ((SGDVector) other).iterator();
                VectorTuple ourTuple;
                VectorTuple otherTuple;

                while (ourItr.hasNext() && otherItr.hasNext()) {
                    ourTuple = ourItr.next();
                    otherTuple = otherItr.next();
                    if (!ourTuple.equals(otherTuple)) {
                        return false;
                    }
                }

                // If one of the iterators still has elements then they are not the same.
                return !(ourItr.hasNext() || otherItr.hasNext());
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(elements);
    }

    /**
     * Adds {@code other} to this vector, producing a new {@link DenseVector}.
     * @param other The vector to add.
     * @return A new {@link DenseVector} where each element value = this.get(i) + other.get(i).
     */
    @Override
    public DenseVector add(SGDVector other) {
        if (other.size() != elements.length) {
            throw new IllegalArgumentException("Can't add two vectors of different dimension, this = " + elements.length + ", other = " + other.size());
        }
        double[] newValues = toArray();
        for (VectorTuple tuple : other) {
            newValues[tuple.index] += tuple.value;
        }
        return new DenseVector(newValues);
    }

    /**
     * Subtracts {@code other} from this vector, producing a new {@link DenseVector}.
     * @param other The vector to subtract.
     * @return A new {@link DenseVector} where each element value = this.get(i) - other.get(i).
     */
    @Override
    public DenseVector subtract(SGDVector other) {
        if (other.size() != elements.length) {
            throw new IllegalArgumentException("Can't subtract two vectors of different dimension, this = " + elements.length + ", other = " + other.size());
        }
        double[] newValues = toArray();
        for (VectorTuple tuple : other) {
            newValues[tuple.index] -= tuple.value;
        }
        return new DenseVector(newValues);
    }

    @Override
    public void intersectAndAddInPlace(Tensor other, DoubleUnaryOperator f) {
        if (other instanceof SGDVector) {
            SGDVector otherVec = (SGDVector) other;
            if (otherVec.size() != elements.length) {
                throw new IllegalArgumentException("Can't intersect two vectors of different dimension, this = " + elements.length + ", other = " + otherVec.size());
            }
            if (otherVec instanceof DenseVector) {
                // If dense, use get as it requires fewer objects
                for (int i = 0; i < elements.length; i++) {
                    elements[i] += f.applyAsDouble(otherVec.get(i));
                }
            } else {
                // Assume sparse
                for (VectorTuple tuple : otherVec) {
                    elements[tuple.index] += f.applyAsDouble(tuple.value);
                }
            }
        } else {
            throw new IllegalArgumentException("Adding a non-Vector to a Vector");
        }
    }

    @Override
    public void hadamardProductInPlace(Tensor other, DoubleUnaryOperator f) {
        if (other instanceof SGDVector) {
            SGDVector otherVec = (SGDVector) other;
            if (otherVec.size() != elements.length) {
                throw new IllegalArgumentException("Can't hadamard product two vectors of different dimension, this = " + elements.length + ", other = " + otherVec.size());
            }
            if (otherVec instanceof DenseVector) {
                // If dense, use get as it requires fewer objects
                for (int i = 0; i < elements.length; i++) {
                    elements[i] *= f.applyAsDouble(otherVec.get(i));
                }
            } else {
                // Assume sparse
                for (VectorTuple tuple : otherVec) {
                    elements[tuple.index] *= f.applyAsDouble(tuple.value);
                }
            }
        } else {
            throw new IllegalArgumentException("Scaling a Vector by a non-Vector");
        }
    }

    @Override
    public void foreachInPlace(DoubleUnaryOperator f) {
        for (int i = 0; i < elements.length; i++) {
            elements[i] = f.applyAsDouble(elements[i]);
        }
    }

    @Override
    public void foreachIndexedInPlace(ToDoubleBiFunction<Integer,Double> f) {
        for (int i = 0; i < elements.length; i++) {
            elements[i] = f.applyAsDouble(i,elements[i]);
        }
    }

    @Override
    public DenseVector scale(double coefficient) {
        DenseVector output = copy();
        output.scaleInPlace(coefficient);
        return output;
    }

    @Override
    public void add(int index, double value) {
        elements[index] += value;
    }

    @Override
    public double dot(SGDVector other) {
        if (other.size() != elements.length) {
            throw new IllegalArgumentException("Can't dot two vectors of different dimension, this = " + elements.length + ", other = " + other.size());
        }
        double score = 0.0;
        if (other instanceof DenseVector) {
            for (int i = 0; i < elements.length; i++) {
                score += get(i) * other.get(i);
            }
        } else {
            // else must be sparse
            for (VectorTuple tuple : other) {
                score += get(tuple.index) * tuple.value;
            }
        }
        return score;
    }

    @Override
    public Matrix outer(SGDVector other) {
        if (other instanceof DenseVector) {
            //Outer product is a DenseMatrix
            DenseVector otherVec = (DenseVector) other;
            double[][] output = new double[elements.length][];
            for (int i = 0; i < elements.length; i++) {
                DenseVector tmp = otherVec.scale(get(i));
                output[i] = tmp.elements;
            }
            return new DenseMatrix(output);
        } else if (other instanceof SparseVector) {
            //Outer product is a DenseSparseMatrix
            SparseVector otherVec = (SparseVector) other;
            SparseVector[] output = new SparseVector[elements.length];
            for (int i = 0; i < elements.length; i++) {
                output[i] = otherVec.scale(get(i));
            }
            return new DenseSparseMatrix(output);
        } else {
            throw new IllegalArgumentException("Invalid vector subclass " + other.getClass().getCanonicalName() + " for input");
        }
    }

    @Override
    public double sum() {
        double sum = 0.0;
        for (int i = 0; i < elements.length; i++) {
            sum += get(i);
        }
        return sum;
    }

    /**
     * Sums this vector, applying the supplied function to each element first.
     * @param f The function to apply to the elements.
     * @return The sum of f(x).
     */
    public double sum(DoubleUnaryOperator f) {
        double sum = 0.0;
        for (int i = 0; i < elements.length; i++) {
            sum += f.applyAsDouble(get(i));
        }
        return sum;
    }

    @Override
    public double twoNorm() {
        double sum = 0.0;
        for (int i = 0; i < elements.length; i++) {
            double value = get(i);
            sum += value * value;
        }
        return Math.sqrt(sum);
    }

    @Override
    public double oneNorm() {
        double sum = 0.0;
        for (int i = 0; i < elements.length; i++) {
            sum += Math.abs(get(i));
        }
        return sum;
    }

    @Override
    public double get(int index) {
        return elements[index];
    }

    @Override
    public void set(int index, double value) {
        elements[index] = value;
    }

    /**
     * Sets all the elements of this vector to be the same as {@code other}.
     * @param other The {@link DenseVector} to copy.
     */
    public void setElements(DenseVector other) {
        for (int i = 0; i < elements.length; i++) {
            elements[i] = other.get(i);
        }
    }

    /**
     * Fills this {@link DenseVector} with {@code value}.
     * @param value The value to store in this vector.
     */
    public void fill(double value) {
        Arrays.fill(elements,value);
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
    public double maxValue() {
        double value = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < elements.length; i++) {
            double tmp = get(i);
            if (tmp > value) {
                value = tmp;
            }
        }
        return value;
    }

    @Override
    public double minValue() {
        double value = Double.POSITIVE_INFINITY;
        for (int i = 0; i < elements.length; i++) {
            double tmp = get(i);
            if (tmp < value) {
                value = tmp;
            }
        }
        return value;
    }

    @Override
    public void normalize(VectorNormalizer normalizer) {
        normalizer.normalizeInPlace(elements);
    }

    /**
     * An optimisation for the exponential normalizer when
     * you already know the normalization constant.
     *
     * Used in the CRF.
     * @param total The normalization constant.
     */
    public void expNormalize(double total) {
        for (int i = 0; i < elements.length; i++) {
            elements[i] = Math.exp(elements[i] - total);
        }
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("DenseVector(size=");
        buffer.append(elements.length);
        buffer.append(",values=[");

        for (int i = 0; i < elements.length; i++) {
            buffer.append(get(i));
            buffer.append(",");
        }
        buffer.setCharAt(buffer.length()-1,']');
        buffer.append(")");

        return buffer.toString();
    }

    @Override
    public double variance(double mean) {
        double variance = 0.0;
        for (int i = 0; i < elements.length; i++) {
            double value = get(i) - mean;
            variance += value * value;
        }
        return variance;
    }

    @Override
    public VectorIterator iterator() {
        return new DenseVectorIterator(this);
    }

    /**
     * Generates a {@link SparseVector} representation from this dense vector, removing all values
     * with absolute value below {@link VectorTuple#DELTA}.
     * @return A {@link SparseVector}.
     */
    public SparseVector sparsify() {
        return sparsify(VectorTuple.DELTA);
    }

    /**
     * Generates a {@link SparseVector} representation from this dense vector, removing all values
     * with absolute value below the supplied tolerance.
     * @param tolerance The threshold below which to set a value to zero.
     * @return A {@link SparseVector}.
     */
    public SparseVector sparsify(double tolerance) {
        ArrayList<Integer> indices = new ArrayList<>();
        ArrayList<Double> values = new ArrayList<>();

        for (int i = 0; i < elements.length; i++) {
            double value = get(i);
            if (Math.abs(value) > tolerance) {
                indices.add(i);
                values.add(value);
            }
        }

        return new SparseVector(elements.length, Util.toPrimitiveInt(indices), Util.toPrimitiveDouble(values));
    }

    /**
     * The l2 or euclidean distance between this vector and the other vector.
     * @param other The other vector.
     * @return The euclidean distance between them.
     */
    @Override
    public double euclideanDistance(SGDVector other) {
        if (other.size() != elements.length) {
            throw new IllegalArgumentException("Can't measure distance of two vectors of different lengths, this = " + elements.length + ", other = " + other.size());
        } else if (other instanceof DenseVector) {
            double score = 0.0;

            for (int i = 0; i < elements.length; i++) {
                double tmp = get(i) - other.get(i);
                score += tmp * tmp;
            }

            return Math.sqrt(score);
        } else if (other instanceof SparseVector) {
            double score = 0.0;

            int i = 0;
            Iterator<VectorTuple> otherItr = other.iterator();
            VectorTuple otherTuple;
            while (i < elements.length && otherItr.hasNext()) {
                otherTuple = otherItr.next();
                //after this loop, either itr is out or tuple.index >= otherTuple.index
                while (i < elements.length && (i < otherTuple.index)) {
                    // as the other vector contains a zero.
                    double value = get(i);
                    score += value*value;
                    i++;
                }
                if (i == otherTuple.index) {
                    double tmp = get(i) - otherTuple.value;
                    score += tmp * tmp;
                    i++;
                }
            }
            for (; i < elements.length; i++) {
                double value = get(i);
                score += value*value;
            }

            return Math.sqrt(score);
        } else {
            throw new IllegalArgumentException("Unknown vector subclass " + other.getClass().getCanonicalName() + " for input");
        }
    }

    /**
     * The l1 or Manhattan distance between this vector and the other vector.
     * @param other The other vector.
     * @return The l1 distance.
     */
    @Override
    public double l1Distance(SGDVector other) {
        if (other.size() != elements.length) {
            throw new IllegalArgumentException("Can't measure distance of two vectors of different lengths, this = " + elements.length + ", other = " + other.size());
        } else if (other instanceof DenseVector) {
            double score = 0.0;

            for (int i = 0; i < elements.length; i++) {
                score += Math.abs(get(i) - other.get(i));
            }

            return score;
        } else if (other instanceof SparseVector) {
            double score = 0.0;

            int i = 0;
            Iterator<VectorTuple> otherItr = other.iterator();
            VectorTuple otherTuple;
            while (i < elements.length && otherItr.hasNext()) {
                otherTuple = otherItr.next();
                //after this loop, either itr is out or tuple.index >= otherTuple.index
                while (i < elements.length && (i < otherTuple.index)) {
                    // as the other vector contains a zero.
                    score += Math.abs(get(i));
                    i++;
                }
                if (i == otherTuple.index) {
                    score += Math.abs(get(i) - otherTuple.value);
                    i++;
                }
            }
            for (; i < elements.length; i++) {
                score += Math.abs(get(i));
            }

            return score;
        } else {
            throw new IllegalArgumentException("Unknown vector subclass " + other.getClass().getCanonicalName() + " for input");
        }
    }

    private static class DenseVectorIterator implements VectorIterator {
        private final DenseVector vector;
        private final VectorTuple tuple;
        private int index;

        public DenseVectorIterator(DenseVector vector) {
            this.vector = vector;
            this.tuple = new VectorTuple();
            this.index = 0;
        }

        @Override
        public boolean hasNext() {
            return index < vector.elements.length;
        }

        @Override
        public VectorTuple next() {
            if (!hasNext()) {
                throw new NoSuchElementException("Off the end of the iterator.");
            }
            tuple.index = index;
            tuple.value = vector.elements[index];
            index++;
            return tuple;
        }

        @Override
        public VectorTuple getReference() {
            return tuple;
        }
    }

}
