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

package org.tribuo.math.la;

import org.tribuo.math.util.VectorNormalizer;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleBiFunction;

/**
 * Interface for 1 dimensional {@link Tensor}s.
 * <p>
 * Vectors have immutable sizes and immutable indices (so {@link SparseVector} can't grow).
 */
public interface SGDVector extends Tensor, Iterable<VectorTuple> {

    /**
     * Applies a {@link ToDoubleBiFunction} elementwise to this {@link SGDVector}.
     * <p>
     * The first argument to the function is the index, the second argument is the current value.
     * @param f The function to apply.
     */
    public void foreachIndexedInPlace(ToDoubleBiFunction<Integer,Double> f);

    /**
     * Returns a deep copy of this vector.
     * @return A copy of this vector.
     */
    @Override
    public SGDVector copy();

    /**
     * Returns the dimensionality of this vector.
     * @return The dimensionality of the vector.
     */
    public int size();

    /**
     * Returns the number of non-zero elements (on construction, an element
     * could be set to zero and it would still remain active).
     * @return The number of non-zero elements.
     */
    public int numActiveElements();

    /**
     * Generates a new vector with each element scaled by {@code coefficient}.
     * @param coefficient The coefficient to scale the elements by.
     * @return A new {@link SGDVector}.
     */
    public SGDVector scale(double coefficient);

    /**
     * Adds {@code value} to the element at {@code index}.
     * @param index The index to update.
     * @param value The value to add.
     */
    public void add(int index, double value);

    /**
     * Adds {@code other} to this vector, producing a new {@link SGDVector}.
     * Adding Dense to Dense/Sparse produces a {@link DenseVector}, adding Sparse to
     * Sparse produces a {@link SparseVector}.
     * @param other The vector to add.
     * @return A new {@link SGDVector} where each element value = this.get(i) + other.get(i).
     */
    public SGDVector add(SGDVector other);

    /**
     * Subtracts {@code other} from this vector, producing a new {@link SGDVector}.
     * Subtracting Dense from Dense/Sparse produces a {@link DenseVector}, subtracting Sparse from
     * Sparse produces a {@link SparseVector}.
     * @param other The vector to subtract.
     * @return A new {@link SGDVector} where each element value = this.get(i) - other.get(i).
     */
    public SGDVector subtract(SGDVector other);

    /**
     * Calculates the dot product between this vector and {@code other}.
     * @param other The other vector.
     * @return The dot product.
     */
    public double dot(SGDVector other);

    /**
     * Generates the matrix representing the outer product between the two vectors.
     * @param other Another {@link SGDVector}
     * @return The outer product {@link Matrix}.
     */
    public Matrix outer(SGDVector other);

    /**
     * Calculates the sum of this vector.
     * @return The sum.
     */
    public double sum();

    /**
     * Calculates the euclidean norm for this vector.
     * @return The euclidean norm.
     */
    @Override
    public double twoNorm();

    /**
     * Calculates the Manhattan norm for this vector.
     * @return The Manhattan norm.
     */
    public double oneNorm();

    /**
     * Gets an element from this vector.
     * @param index The index of the element.
     * @return The value at that index.
     */
    public double get(int index);

    /**
     * Sets the {@code index} to the {@code value}.
     * @param index The index to set.
     * @param value The value to set it to.
     */
    public void set(int index, double value);

    /**
     * Returns the index of the maximum value. Requires probing the array.
     * @return The index of the maximum value.
     */
    public int indexOfMax();

    /**
     * Returns the maximum value. Requires probing the array.
     * @return The maximum value.
     */
    public double maxValue();

    /**
     * Returns the minimum value. Requires probing the array.
     * @return The minimum value.
     */
    public double minValue();

    /**
     * Normalizes the vector using the supplied vector normalizer.
     * @param normalizer The kind of normalization to apply.
     */
    public void normalize(VectorNormalizer normalizer);

    /**
     * Reduces the vector, applying the transformation to every value (including the implicit zeros)
     * and reducing the output by applying the supplied reduction operator (where the right argument is
     * the current reduction value, and the left argument is the transformed value). The reduction
     * operation is seeded with the initial value.
     * @param initial The initial value for the reduction.
     * @param transform The transformation operator.
     * @param reduction The reduction operator.
     * @return The reduction of this vector.
     */
    public double reduce(double initial, DoubleUnaryOperator transform, DoubleBinaryOperator reduction);

    /**
     * Synonym for euclideanDistance.
     * @param other The other vector.
     * @return The l2 norm of the difference between the two vectors.
     */
    default public double l2Distance(SGDVector other) {
        return euclideanDistance(other);
    }

    /**
     * The l2 or euclidean distance between this vector and the other vector.
     * @param other The other vector.
     * @return The euclidean distance between them.
     */
    public double euclideanDistance(SGDVector other);

    /**
     * The l1 or Manhattan distance between this vector and the other vector.
     * @param other The other vector.
     * @return The l1 distance.
     */
    public double l1Distance(SGDVector other);

    /**
     * Calculates the cosine distance of two vectors.
     * 1 - cos(x,y)
     * @param other The other vector.
     * @return 1 - cosine similarity (this,other)
     */
    default public double cosineDistance(SGDVector other) {
        return 1 - cosineSimilarity(other);
    }

    /**
     * Calculates the cosine similarity of two vectors.
     * cos(x,y) = dot(x,y) / (norm(x) * norm(y))
     * @param other The other vector.
     * @return cosine similarity (this,other)
     */
    default public double cosineSimilarity(SGDVector other) {
        double numerator = dot(other);
        double output = 0.0;
        if (numerator != 0.0) {
            output = numerator / (twoNorm() * other.twoNorm());
        }
        return output;
    }

    /**
     * Calculates the variance of this vector.
     * @return The variance of the vector.
     */
    default public double variance() {
        double mean = sum() / size();
        return variance(mean);
    }

    /**
     * Calculates the variance of this vector based on the supplied mean.
     * @param mean The mean of the vector.
     * @return The variance of the vector.
     */
    public double variance(double mean);

    /**
     * Returns an array containing all the values in the vector (including any implicit zeros).
     * @return An array copy.
     */
    public double[] toArray();
}
