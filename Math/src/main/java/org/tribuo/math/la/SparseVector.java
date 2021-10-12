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

import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.util.IntDoublePair;
import org.tribuo.util.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleBiFunction;
import java.util.stream.Collectors;

/**
 * A sparse vector. Stored as a sorted array of indices and an array of values.
 * <p>
 * Uses binary search to look up a specific index, so it's usually faster to
 * use the iterator to iterate the values.
 * <p>
 * This vector has immutable indices. It cannot get new indices after construction,
 * and will throw {@link IllegalArgumentException} if such an operation is tried.
 */
public class SparseVector implements SGDVector {
    private static final long serialVersionUID = 1L;

    private final int[] shape;
    protected final int[] indices;
    protected final double[] values;
    private final int size;

    /**
     * Creates an empty sparse vector.
     * @param size The dimension.
     */
    SparseVector(int size) {
        this.indices = new int[0];
        this.values = new double[0];
        this.size = size;
        this.shape = new int[]{size};
    }

    /**
     * Used internally for performance.
     * Does not defensively copy the input, nor check it's sorted.
     * <p>
     * @param size The dimension of this vector.
     * @param indices The indices.
     * @param values The values.
     */
    SparseVector(int size, int[] indices, double[] values) {
        this.size = size;
        this.shape = new int[]{size};
        this.indices = indices;
        this.values = values;
    }

    /**
     * Returns a deep copy of the supplied sparse vector.
     * <p>
     * Copies the value by iterating it's VectorTuple.
     * @param other The SparseVector to copy.
     */
    private SparseVector(SparseVector other) {
        this.size = other.size;
        int numActiveElements = other.numActiveElements();
        this.indices = new int[numActiveElements];
        this.values = new double[numActiveElements];

        int i = 0;
        for (VectorTuple tuple : other) {
            indices[i] = tuple.index;
            values[i] = tuple.value;
            i++;
        }
        this.shape = new int[]{size};
    }

    /**
     * Creates a sparse vector of the specified size, with the supplied value at each of the indices.
     * @param size The vector size.
     * @param indices The active indices.
     * @param value The value for those indices.
     */
    public SparseVector(int size, int[] indices, double value) {
        this.indices = Arrays.copyOf(indices,indices.length);
        this.values = new double[indices.length];
        Arrays.fill(this.values,value);
        this.size = size;
        this.shape = new int[]{size};
    }

    /**
     * Builds a {@link SparseVector} from an {@link Example}.
     * <p>
     * Used in training and inference.
     * <p>
     * Throws {@link IllegalArgumentException} if the Example contains NaN-valued features.
     * @param example     The example to convert.
     * @param featureInfo The feature information, used to calculate the dimension of this SparseVector.
     * @param addBias     Add a bias feature.
     * @param <T>         The type parameter of the {@code example}.
     * @return A SparseVector representing the example's features.
     */
    public static <T extends Output<T>> SparseVector createSparseVector(Example<T> example, ImmutableFeatureMap featureInfo, boolean addBias) {
        int size;
        int numFeatures = example.size();
        if (addBias) {
            size = featureInfo.size() + 1;
            numFeatures++;
        } else {
            size = featureInfo.size();
        }
        int[] tmpIndices = new int[numFeatures];
        double[] tmpValues = new double[numFeatures];
        int i = 0;
        int prevIdx = -1;
        for (Feature f : example) {
            int index = featureInfo.getID(f.getName());
            if (index > prevIdx){
                prevIdx = index;
                tmpIndices[i] = index;
                tmpValues[i] = f.getValue();
                if (Double.isNaN(tmpValues[i])) {
                    throw new IllegalArgumentException("Example contained a NaN feature, " + f.toString());
                }
                i++;
            } else if (index > -1) {
                //
                // Collision, deal with it.
                int collisionIdx = Arrays.binarySearch(tmpIndices,0,i,index);
                if (collisionIdx < 0) {
                    //
                    // Collision but not present in tmpIndices
                    // move data and bump i
                    collisionIdx = - (collisionIdx + 1);
                    System.arraycopy(tmpIndices,collisionIdx,tmpIndices,collisionIdx+1,i-collisionIdx);
                    System.arraycopy(tmpValues,collisionIdx,tmpValues,collisionIdx+1,i-collisionIdx);
                    tmpIndices[collisionIdx] = index;
                    tmpValues[collisionIdx] = f.getValue();
                    if (Double.isNaN(tmpValues[collisionIdx])) {
                        throw new IllegalArgumentException("Example contained a NaN feature, " + f.toString());
                    }
                    i++;
                } else {
                    //
                    // Collision present in tmpIndices
                    // add the values.
                    tmpValues[collisionIdx] += f.getValue();
                    if (Double.isNaN(tmpValues[collisionIdx])) {
                        throw new IllegalArgumentException("Example contained a NaN feature, " + f.toString());
                    }
                }
            }
        }
        if (addBias) {
            tmpIndices[i] = size - 1;
            tmpValues[i] = 1.0;
            i++;
        }
        return new SparseVector(size,Arrays.copyOf(tmpIndices,i),Arrays.copyOf(tmpValues,i));
    }

    /**
     * Defensively copies the input, and checks that the indices are sorted. If not,
     * it sorts them.
     * <p>
     * Throws {@link IllegalArgumentException} if the arrays are not the same length, or if size is less than
     * the max index.
     * @param dimension The dimension of this vector.
     * @param indices The indices of the non-zero elements.
     * @param values The values of the non-zero elements.
     * @return A SparseVector encapsulating the indices and values.
     */
    public static SparseVector createSparseVector(int dimension, int[] indices, double[] values) {
        if (indices.length != values.length) {
            throw new IllegalArgumentException("Indices and values must be the same length, found indices.length = " + indices.length + " and values.length = " + values.length);
        } else if (indices.length == 0) {
            return new SparseVector(dimension,indices,values);
        } else {
            IntDoublePair[] pairArray = new IntDoublePair[indices.length];
            for (int i = 0; i < pairArray.length; i++) {
                pairArray[i] = new IntDoublePair(indices[i], values[i]);
            }
            Arrays.sort(pairArray, IntDoublePair.pairIndexComparator());
            int[] newIndices = new int[indices.length];
            double[] newValues = new double[values.length];
            for (int i = 0; i < pairArray.length; i++) {
                newIndices[i] = pairArray[i].index;
                newValues[i] = pairArray[i].value;
            }
            if (dimension < newIndices[newIndices.length - 1]) {
                throw new IllegalArgumentException("Number of dimensions is less than the maximum index, dimensions = " + dimension + ", max index = " + newIndices[newIndices.length - 1]);
            }
            return new SparseVector(dimension, newIndices, newValues);
        }
    }

    /**
     * Builds a SparseVector from a map.
     * <p>
     * Throws {@link IllegalArgumentException} if dimension is less than the max index.
     * @param dimension The dimension of this vector.
     * @param indexMap The map from indices to values.
     * @return A SparseVector.
     */
    public static SparseVector createSparseVector(int dimension, Map<Integer, Double> indexMap) {
        if (indexMap.isEmpty()) {
            return new SparseVector(dimension,new int[0],new double[0]);
        } else {
            List<Map.Entry<Integer, Double>> sortedEntries = indexMap.entrySet()
                    .stream().sorted(Map.Entry.comparingByKey())
                    .collect(Collectors.toList());

            int[] indices = new int[sortedEntries.size()];
            double[] values = new double[sortedEntries.size()];
            for (int i = 0; i < sortedEntries.size(); i++) {
                indices[i] = sortedEntries.get(i).getKey();
                values[i] = sortedEntries.get(i).getValue();
            }
            if (dimension < indices[indices.length - 1]) {
                throw new IllegalArgumentException("Number of dimensions is less than the maximum index, dimensions = " + dimension + ", max index = " + indices[indices.length - 1]);
            }
            return new SparseVector(dimension, indices, values);
        }
    }

    @Override
    public SparseVector copy() {
        return new SparseVector(this);
    }

    @Override
    public int[] getShape() {
        return shape;
    }

    @Override
    public Tensor reshape(int[] newShape) {
        throw new UnsupportedOperationException("Reshape not supported on sparse Tensors.");
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int numActiveElements() {
        return values.length;
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
            Iterator<VectorTuple> ourItr = iterator();
            Iterator<VectorTuple> otherItr = ((SGDVector)other).iterator();
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
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(size);
        result = 31 * result + Arrays.hashCode(indices);
        result = 31 * result + Arrays.hashCode(values);
        return result;
    }

    /**
     * Adds {@code other} to this vector, producing a new {@link SGDVector}.
     * If {@code other} is a {@link SparseVector} then the returned vector is also
     * a {@link SparseVector} otherwise it's a {@link DenseVector}.
     * @param other The vector to add.
     * @return A new {@link SGDVector} where each element value = this.get(i) + other.get(i).
     */
    @Override
    public SGDVector add(SGDVector other) {
        if (other.size() != size) {
            throw new IllegalArgumentException("Can't add two vectors of different dimension, this = " + size + ", other = " + other.size());
        }
        if (other instanceof DenseVector) {
            return other.add(this);
        } else if (other instanceof SparseVector) {
            Map<Integer, Double> values = new HashMap<>();
            for (VectorTuple tuple : this) {
                values.put(tuple.index, tuple.value);
            }
            for (VectorTuple tuple : other) {
                values.merge(tuple.index, tuple.value, Double::sum);
            }
            return createSparseVector(size, values);
        } else {
            throw new IllegalArgumentException("Vector other is not dense or sparse.");
        }
    }

    /**
     * Subtracts {@code other} from this vector, producing a new {@link SGDVector}.
     * If {@code other} is a {@link SparseVector} then the returned vector is also
     * a {@link SparseVector} otherwise it's a {@link DenseVector}.
     * @param other The vector to subtract.
     * @return A new {@link SGDVector} where each element value = this.get(i) - other.get(i).
     */
    @Override
    public SGDVector subtract(SGDVector other) {
        if (other.size() != size) {
            throw new IllegalArgumentException("Can't subtract two vectors of different dimension, this = " + size + ", other = " + other.size());
        }
        if (other instanceof DenseVector) {
            DenseVector output = ((DenseVector)other).copy();
            for (VectorTuple tuple : this) {
                output.set(tuple.index,tuple.value-output.get(tuple.index));
            }
            return output;
        } else if (other instanceof SparseVector) {
            Map<Integer, Double> values = new HashMap<>();
            for (VectorTuple tuple : this) {
                values.put(tuple.index, tuple.value);
            }
            for (VectorTuple tuple : other) {
                values.merge(tuple.index, -tuple.value, Double::sum);
            }
            return createSparseVector(size, values);
        } else {
            throw new IllegalArgumentException("Vector other is not dense or sparse.");
        }
    }

    @Override
    public void intersectAndAddInPlace(Tensor other, DoubleUnaryOperator f) {
        if (other instanceof SparseVector) {
            SparseVector otherVec = (SparseVector) other;
            if (otherVec.size() != size) {
                throw new IllegalArgumentException("Can't intersect two vectors of different dimension, this = " + size + ", other = " + otherVec.size());
            } else if (otherVec.numActiveElements() > 0) {
                int i = 0;
                Iterator<VectorTuple> otherItr = otherVec.iterator();
                VectorTuple tuple = otherItr.next();
                while (i < (indices.length-1) && otherItr.hasNext()) {
                    if (indices[i] == tuple.index) {
                        values[i] += f.applyAsDouble(tuple.value);
                        i++;
                        tuple = otherItr.next();
                    } else if (indices[i] < tuple.index) {
                        i++;
                    } else {
                        tuple = otherItr.next();
                    }
                }
                for (; i < indices.length-1; i++) {
                    if (indices[i] == tuple.index) {
                        values[i] += f.applyAsDouble(tuple.value);
                    }
                }
                while (otherItr.hasNext()) {
                    if (indices[i] == tuple.index) {
                        values[i] += f.applyAsDouble(tuple.value);
                    }
                    tuple = otherItr.next();
                }
                if (indices[i] == tuple.index) {
                    values[i] += f.applyAsDouble(tuple.value);
                }
            }
        } else if (other instanceof DenseVector) {
            DenseVector otherVec = (DenseVector) other;
            if (otherVec.size() != size) {
                throw new IllegalArgumentException("Can't intersect two vectors of different dimension, this = " + size + ", other = " + otherVec.size());
            }
            for (int i = 0; i < indices.length; i++) {
                values[i] += f.applyAsDouble(otherVec.get(indices[i]));
            }
        } else {
            throw new IllegalStateException("Unknown Tensor subclass " + other.getClass().getCanonicalName() + " for input");
        }
    }

    @Override
    public void hadamardProductInPlace(Tensor other, DoubleUnaryOperator f) {
        if (other instanceof SparseVector) {
            SparseVector otherVec = (SparseVector) other;
            if (otherVec.size() != size) {
                throw new IllegalArgumentException("Can't hadamard product two vectors of different dimension, this = " + size + ", other = " + otherVec.size());
            } else if (otherVec.numActiveElements() > 0) {
                int i = 0;
                Iterator<VectorTuple> otherItr = otherVec.iterator();
                VectorTuple tuple = otherItr.next();
                while (i < (indices.length-1) && otherItr.hasNext()) {
                    if (indices[i] == tuple.index) {
                        values[i] *= f.applyAsDouble(tuple.value);
                        i++;
                        tuple = otherItr.next();
                    } else if (indices[i] < tuple.index) {
                        i++;
                    } else {
                        tuple = otherItr.next();
                    }
                }
                for (; i < indices.length-1; i++) {
                    if (indices[i] == tuple.index) {
                        values[i] *= f.applyAsDouble(tuple.value);
                    }
                }
                while (otherItr.hasNext()) {
                    if (indices[i] == tuple.index) {
                        values[i] *= f.applyAsDouble(tuple.value);
                    }
                    tuple = otherItr.next();
                }
                if (indices[i] == tuple.index) {
                    values[i] *= f.applyAsDouble(tuple.value);
                }
            }
        } else if (other instanceof DenseVector) {
            DenseVector otherVec = (DenseVector) other;
            if (otherVec.size() != size) {
                throw new IllegalArgumentException("Can't hadamard product two vectors of different dimension, this = " + size + ", other = " + otherVec.size());
            }
            for (int i = 0; i < indices.length; i++) {
                values[i] *= f.applyAsDouble(otherVec.get(indices[i]));
            }
        } else {
            throw new IllegalArgumentException("Invalid Tensor subclass " + other.getClass().getCanonicalName() + " for input");
        }
    }

    /**
     * Applies a {@link DoubleUnaryOperator} elementwise to this {@link SGDVector}.
     * <p>
     * Only applies the function to the elements which are present.
     * <p>
     * If you need to operate over the whole vector then densify it first.
     * @param f The function to apply.
     */
    @Override
    public void foreachInPlace(DoubleUnaryOperator f) {
        for (int i = 0; i < values.length; i++) {
            values[i] = f.applyAsDouble(values[i]);
        }
    }

    /**
     * Applies a {@link ToDoubleBiFunction} elementwise to this {@link SGDVector}.
     * <p>
     * The first argument to the function is the index, the second argument is the current value.
     * <p>
     * Only applies the function to the elements which are present.
     * <p>
     * If you need to operate over the whole vector then densify it first.
     * @param f The function to apply.
     */
    @Override
    public void foreachIndexedInPlace(ToDoubleBiFunction<Integer,Double> f) {
        for (int i = 0; i < values.length; i++) {
            values[i] = f.applyAsDouble(indices[i],values[i]);
        }
    }

    @Override
    public SparseVector scale(double coefficient) {
        double[] newValues = Arrays.copyOf(values, values.length);
        for (int i = 0; i < values.length; i++) {
            newValues[i] *= coefficient;
        }
        return new SparseVector(size, Arrays.copyOf(indices, indices.length), newValues);
    }

    @Override
    public void add(int index, double value) {
        int foundIndex = Arrays.binarySearch(indices, index);
        if (foundIndex < 0) {
            throw new IllegalArgumentException("SparseVector cannot have new elements added.");
        } else {
            values[foundIndex] += value;
        }
    }

    @Override
    public double dot(SGDVector other) {
        if (other.size() != size) {
            throw new IllegalArgumentException("Can't dot two vectors of different lengths, this = " + size + ", other = " + other.size());
        } else if (other instanceof SparseVector) {
            double score = 0.0;

            // If there are elements, calculate the dot product.
            if ((other.numActiveElements() != 0) && (indices.length != 0)) {
                Iterator<VectorTuple> itr = iterator();
                Iterator<VectorTuple> otherItr = other.iterator();
                VectorTuple tuple = itr.next();
                VectorTuple otherTuple = otherItr.next();
                while (itr.hasNext() && otherItr.hasNext()) {
                    if (tuple.index == otherTuple.index) {
                        score += tuple.value * otherTuple.value;
                        tuple = itr.next();
                        otherTuple = otherItr.next();
                    } else if (tuple.index < otherTuple.index) {
                        tuple = itr.next();
                    } else {
                        otherTuple = otherItr.next();
                    }
                }
                while (itr.hasNext()) {
                    if (tuple.index == otherTuple.index) {
                        score += tuple.value * otherTuple.value;
                    }
                    tuple = itr.next();
                }
                while (otherItr.hasNext()) {
                    if (tuple.index == otherTuple.index) {
                        score += tuple.value * otherTuple.value;
                    }
                    otherTuple = otherItr.next();
                }
                if (tuple.index == otherTuple.index) {
                    score += tuple.value * otherTuple.value;
                }
            }

            return score;
        } else if (other instanceof DenseVector) {
            double score = 0.0;

            for (int i = 0; i < indices.length; i++) {
                score += other.get(indices[i]) * values[i];
            }

            return score;
        } else {
            throw new IllegalArgumentException("Unknown vector subclass " + other.getClass().getCanonicalName() + " for input");
        }
    }

    /**
     * This generates the outer product when dotted with another {@link SparseVector}.
     * <p>
     * It throws an {@link IllegalArgumentException} if used with a {@link DenseVector}.
     *
     * @param other A vector.
     * @return A {@link DenseSparseMatrix} representing the outer product.
     */
    @Override
    public Matrix outer(SGDVector other) {
        if (other instanceof SparseVector) {
            //This horrible mess is why there should be a sparse-sparse matrix type.
            SparseVector otherVec = (SparseVector) other;
            SparseVector[] output = new SparseVector[size];
            int i = 0;
            for (VectorTuple tuple : this) {
                while (i < tuple.index) {
                    output[i] = new SparseVector(other.size(), new int[0], new double[0]);
                    i++;
                }
                output[tuple.index] = otherVec.scale(tuple.value);
                i++;
            }
            while (i < output.length) {
                output[i] = new SparseVector(other.size(), new int[0], new double[0]);
                i++;
            }
            //TODO this is suboptimal if there are lots of missing rows.
            return new DenseSparseMatrix(output);
        } else if (other instanceof DenseVector) {
            //Outer product is a DenseMatrix because DenseSparseMatrix is the wrong way around.
            DenseVector otherVec = (DenseVector) other;
            int otherSize = otherVec.size();
            double[][] output = new double[size][];
            int i = 0;
            for (VectorTuple tuple : this) {
                while (i < tuple.index) {
                    output[i] = new double[otherSize];
                    i++;
                }
                DenseVector tmp = otherVec.scale(tuple.value);
                output[tuple.index] = tmp.elements;
                i++;
            }
            while (i < output.length) {
                output[i] = new double[otherSize];
                i++;
            }
            //TODO this is also suboptimal if there are lots of missing rows.
            return new DenseMatrix(output);
        } else {
            throw new IllegalArgumentException("Unknown vector subclass " + other.getClass().getCanonicalName() + " for input");
        }
    }

    @Override
    public double sum() {
        double sum = 0.0;
        for (int i = 0; i < values.length; i++) {
            sum += values[i];
        }
        return sum;
    }

    @Override
    public double twoNorm() {
        double sum = 0.0;
        for (int i = 0; i < values.length; i++) {
            sum += values[i] * values[i];
        }
        return Math.sqrt(sum);
    }

    @Override
    public double oneNorm() {
        double sum = 0.0;
        for (int i = 0; i < values.length; i++) {
            sum += Math.abs(values[i]);
        }
        return sum;
    }

    @Override
    public double get(int index) {
        int foundIndex = Arrays.binarySearch(indices, index);
        if (foundIndex < 0) {
            return 0;
        } else {
            return values[foundIndex];
        }
    }

    @Override
    public void set(int index, double value) {
        int foundIndex = Arrays.binarySearch(indices, index);
        if (foundIndex < 0) {
            throw new IllegalArgumentException("SparseVector cannot have new elements added.");
        } else {
            values[foundIndex] = value;
        }
    }

    @Override
    public int indexOfMax() {
        int index = 0;
        double value = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            double tmp = values[i];
            if (tmp > value) {
                index = i;
                value = tmp;
            }
        }
        return indices[index];
    }

    @Override
    public double maxValue() {
        double value = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            double tmp = values[i];
            if (tmp > value) {
                value = tmp;
            }
        }
        return value;
    }

    @Override
    public double minValue() {
        double value = Double.POSITIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            double tmp = values[i];
            if (tmp < value) {
                value = tmp;
            }
        }
        return value;
    }

    /**
     * Generates an array of the indices that are active in this vector
     * but are not present in {@code other}.
     *
     * @param other The vector to compare.
     * @return An array of indices that are active only in this vector.
     */
    public int[] difference(SparseVector other) {
        List<Integer> diffIndicesList = new ArrayList<>();

        if (other.numActiveElements() == 0) {
            return Arrays.copyOf(indices,indices.length);
        } else if (indices.length == 0) {
            return new int[0];
        } else {
            Iterator<VectorTuple> itr = iterator();
            Iterator<VectorTuple> otherItr = other.iterator();
            VectorTuple tuple = itr.next();
            VectorTuple otherTuple = otherItr.next();
            while (itr.hasNext() && otherItr.hasNext()) {
                if (tuple.index == otherTuple.index) {
                    tuple = itr.next();
                    otherTuple = otherItr.next();
                } else if (tuple.index < otherTuple.index) {
                    diffIndicesList.add(tuple.index);
                    tuple = itr.next();
                } else {
                    otherTuple = otherItr.next();
                }
            }
            while (itr.hasNext()) {
                if (tuple.index != otherTuple.index) {
                    diffIndicesList.add(tuple.index);
                }
                tuple = itr.next();
            }
            while (otherItr.hasNext()) {
                if (tuple.index == otherTuple.index) {
                    break; // break out of loop as we've found the last value.
                }
                otherTuple = otherItr.next();
            }
            if (tuple.index != otherTuple.index) {
                diffIndicesList.add(tuple.index);
            }
        }

        return Util.toPrimitiveInt(diffIndicesList);
    }

    /**
     * Generates an array of the indices that are active in both this
     * vector and {@code other}
     *
     * @param other The vector to intersect.
     * @return An array of indices that are active in both vectors.
     */
    public int[] intersection(SparseVector other) {
        List<Integer> diffIndicesList = new ArrayList<>();

        Iterator<VectorTuple> itr = iterator();
        Iterator<VectorTuple> otherItr = other.iterator();
        if (itr.hasNext() && otherItr.hasNext()) {
            VectorTuple tuple = itr.next();
            VectorTuple otherTuple = otherItr.next();
            while (itr.hasNext() && otherItr.hasNext()) {
                if (tuple.index == otherTuple.index) {
                    diffIndicesList.add(tuple.index);
                    tuple = itr.next();
                    otherTuple = otherItr.next();
                } else if (tuple.index < otherTuple.index) {
                    tuple = itr.next();
                } else {
                    otherTuple = otherItr.next();
                }
            }
            while (itr.hasNext()) {
                if (tuple.index == otherTuple.index) {
                    diffIndicesList.add(tuple.index);
                }
                tuple = itr.next();
            }
            while (otherItr.hasNext()) {
                if (tuple.index == otherTuple.index) {
                    diffIndicesList.add(tuple.index);
                }
                otherTuple = otherItr.next();
            }
            if (tuple.index == otherTuple.index) {
                diffIndicesList.add(tuple.index);
            }
        }

        return Util.toPrimitiveInt(diffIndicesList);
    }


    @Override
    public void normalize(VectorNormalizer normalizer) {
        throw new UnsupportedOperationException("Can't normalize a sparse array");
    }

    @Override
    public double reduce(double initial, DoubleUnaryOperator transform, DoubleBinaryOperator reduction) {
        double output = initial;

        double transformedZero = transform.applyAsDouble(0.0f);

        int i = 0;
        for (VectorTuple tuple : this) {
            while (i < tuple.index) {
                output = reduction.applyAsDouble(transformedZero,output);
                i++;
            }
            double transformed = transform.applyAsDouble(tuple.value);
            output = reduction.applyAsDouble(transformed,output);
            i++;
        }
        while (i < size) {
            output = reduction.applyAsDouble(transformedZero,output);
            i++;
        }

        return output;
    }

    @Override
    public double euclideanDistance(SGDVector other) {
        return distance(other,(double a) -> a*a, Math::sqrt);
    }

    @Override
    public double l1Distance(SGDVector other) {
        return distance(other,Math::abs,DoubleUnaryOperator.identity());
    }

    /**
     * Computes the distance between this vector and the other vector.
     * @param other The other vector.
     * @param transformFunc The transformation function to apply to each paired dimension difference.
     * @param normalizeFunc The normalization to apply after summing the transformed differences.
     * @return The distance between the two vectors.
     */
    public double distance(SGDVector other, DoubleUnaryOperator transformFunc, DoubleUnaryOperator normalizeFunc) {
        if (other.size() != size) {
            throw new IllegalArgumentException("Can't measure the distance between two vectors of different lengths, this = " + size + ", other = " + other.size());
        }
        double score = 0.0;

        if ((other.numActiveElements() != 0) && (indices.length != 0)){
            Iterator<VectorTuple> itr = iterator();
            Iterator<VectorTuple> otherItr = other.iterator();
            VectorTuple tuple = itr.next();
            VectorTuple otherTuple = otherItr.next();
            while (itr.hasNext() && otherItr.hasNext()) {
                if (tuple.index == otherTuple.index) {
                    score += transformFunc.applyAsDouble(tuple.value - otherTuple.value);
                    tuple = itr.next();
                    otherTuple = otherItr.next();
                } else if (tuple.index < otherTuple.index) {
                    score += transformFunc.applyAsDouble(tuple.value);
                    tuple = itr.next();
                } else {
                    score += transformFunc.applyAsDouble(otherTuple.value);
                    otherTuple = otherItr.next();
                }
            }
            while (itr.hasNext()) {
                if (tuple.index == otherTuple.index) {
                    score += transformFunc.applyAsDouble(tuple.value - otherTuple.value);
                    otherTuple = new VectorTuple(); // Consumed this value, replace with sentinel
                } else {
                    score += transformFunc.applyAsDouble(tuple.value);
                }
                tuple = itr.next();
            }
            while (otherItr.hasNext()) {
                if (tuple.index == otherTuple.index) {
                    score += transformFunc.applyAsDouble(tuple.value - otherTuple.value);
                    tuple = new VectorTuple(); // Consumed this value, replace with sentinel
                } else {
                    score += transformFunc.applyAsDouble(otherTuple.value);
                }
                otherTuple = otherItr.next();
            }
            if (tuple.index == otherTuple.index) {
                score += transformFunc.applyAsDouble(tuple.value - otherTuple.value);
            } else {
                if (tuple.index != -1) {
                    score += transformFunc.applyAsDouble(tuple.value);
                }
                if (otherTuple.index != -1) {
                    score += transformFunc.applyAsDouble(otherTuple.value);
                }
            }
        } else if (indices.length != 0) {
            for (VectorTuple tuple : this) {
                score += transformFunc.applyAsDouble(tuple.value);
            }
        } else {
            for (VectorTuple tuple : other) {
                score += transformFunc.applyAsDouble(tuple.value);
            }
        }

        return normalizeFunc.applyAsDouble(score);
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("SparseVector(size=");
        buffer.append(size);
        buffer.append(",tuples=");

        for (int i = 0; i < indices.length; i++) {
            buffer.append("[");
            buffer.append(indices[i]);
            buffer.append(",");
            buffer.append(values[i]);
            buffer.append("],");
        }
        buffer.setCharAt(buffer.length() - 1, ')');

        return buffer.toString();
    }

    /**
     * Returns a dense vector copying this sparse vector.
     * @return A dense copy of this vector.
     */
    public DenseVector densify() {
        return new DenseVector(toArray());
    }

    /**
     * Generates a dense array copy of this SparseVector.
     * @return A dense array containing this vector along with the implicit zeros.
     */
    @Deprecated
    public double[] toDenseArray() {
        return toArray();
    }

    @Override
    public double[] toArray() {
        double[] output = new double[size];
        for (int i = 0; i < values.length; i++) {
            output[indices[i]] = values[i];
        }
        return output;
    }

    @Override
    public double variance(double mean) {
        double variance = 0.0;
        for (int i = 0; i < values.length; i++) {
            variance += (values[i] - mean) * (values[i] - mean);
        }
        variance += (size - values.length) * mean * mean;
        return variance;
    }

    @Override
    public VectorIterator iterator() {
        return new SparseVectorIterator(this);
    }

    private static class SparseVectorIterator implements VectorIterator {
        private final SparseVector vector;
        private final VectorTuple tuple;
        private int index;

        public SparseVectorIterator(SparseVector vector) {
            this.vector = vector;
            this.tuple = new VectorTuple();
            this.index = 0;
        }

        @Override
        public boolean hasNext() {
            return index < vector.indices.length;
        }

        @Override
        public VectorTuple next() {
            if (!hasNext()) {
                throw new NoSuchElementException("Off the end of the iterator.");
            }
            tuple.index = vector.indices[index];
            tuple.value = vector.values[index];
            index++;
            return tuple;
        }

        @Override
        public VectorTuple getReference() {
            return tuple;
        }
    }

    /**
     * Transposes an array of sparse vectors from row-major to column-major or
     * vice versa.
     * @param input Input sparse vectors.
     * @return A column-major array of SparseVectors.
     */
    public static SparseVector[] transpose(SparseVector[] input) {
        int firstDimension = input.length;
        int secondDimension = input[0].size;

        ArrayList<ArrayList<Integer>> indices = new ArrayList<>();
        ArrayList<ArrayList<Double>> values = new ArrayList<>();

        for (int i = 0; i < secondDimension; i++) {
            indices.add(new ArrayList<>());
            values.add(new ArrayList<>());
        }

        for (int i = 0; i < firstDimension; i++) {
            for (VectorTuple f : input[i]) {
                indices.get(f.index).add(i);
                values.get(f.index).add(f.value);
            }
        }

        SparseVector[] output = new SparseVector[secondDimension];

        for (int i = 0; i < secondDimension; i++) {
            output[i] = new SparseVector(firstDimension,Util.toPrimitiveInt(indices.get(i)),Util.toPrimitiveDouble(values.get(i)));
        }

        return output;
    }

    /**
     * Converts a dataset of row-major examples into an array of column-major
     * sparse vectors.
     * @param dataset Input dataset.
     * @param <T> The type of the dataset.
     * @return A column-major array of SparseVectors.
     */
    public static <T extends Output<T>> SparseVector[] transpose(Dataset<T> dataset) {
        ImmutableFeatureMap fMap = dataset.getFeatureIDMap();
        return transpose(dataset,fMap);
    }

    /**
     * Converts a dataset of row-major examples into an array of column-major
     * sparse vectors.
     * @param dataset Input dataset.
     * @param fMap The feature map to use. If it's different to the feature map used by the dataset then behaviour is undefined.
     * @param <T> The type of the dataset.
     * @return A column-major array of SparseVectors.
     */
    public static <T extends Output<T>> SparseVector[] transpose(Dataset<T> dataset, ImmutableFeatureMap fMap) {
        if (dataset.getFeatureMap().size() != fMap.size()) {
            throw new IllegalArgumentException(
                    "The dataset's internal feature map and the supplied feature map have different sizes. dataset = "
                    + dataset.getFeatureMap().size() + ", fMap = " + fMap.size());
        }
        int numExamples = dataset.size();
        int numFeatures = fMap.size();

        ArrayList<ArrayList<Integer>> indices = new ArrayList<>();
        ArrayList<ArrayList<Double>> values = new ArrayList<>();

        for (int i = 0; i < numFeatures; i++) {
            indices.add(new ArrayList<>());
            values.add(new ArrayList<>());
        }

        int j = 0;
        for (Example<T> e : dataset) {
            for (Feature f : e) {
                int index = fMap.getID(f.getName());
                indices.get(index).add(j);
                values.get(index).add(f.getValue());
            }
            j++;
        }

        SparseVector[] output = new SparseVector[numFeatures];

        for (int i = 0; i < fMap.size(); i++) {
            output[i] = new SparseVector(numExamples,Util.toPrimitiveInt(indices.get(i)),Util.toPrimitiveDouble(values.get(i)));
        }

        return output;
    }
}

