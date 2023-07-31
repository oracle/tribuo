/*
 * Copyright (c) 2015, 2023, Oracle and/or its affiliates. All rights reserved.
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

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Interface for 2 dimensional {@link Tensor}s.
 * <p>
 * Matrices have immutable sizes and immutable indices (so {@link DenseSparseMatrix} can't grow).
 */
public interface Matrix extends Tensor, Iterable<MatrixTuple> {

    /**
     * Copies the matrix.
     * @return A copy of the matrix.
     */
    @Override
    public Matrix copy();

    /**
     * Gets an element from this {@link Matrix}.
     * @param i The index for the first dimension.
     * @param j The index for the second dimension.
     * @return The value at matrix[i][j].
     */
    public double get(int i, int j);

    /**
     * Sets an element at the supplied location.
     * @param i The index for the first dimension.
     * @param j The index for the second dimension.
     * @param value The value to be stored at matrix[i][j].
     */
    public void set(int i, int j, double value);

    /**
     * Adds the argument value to the value at the supplied index.
     * @param i The index for the first dimension.
     * @param j The index for the second dimension.
     * @param value The value to add.
     */
    public void add(int i, int j, double value);

    /**
     * The size of the first dimension.
     * @return The size of the first dimension.
     */
    public int getDimension1Size();

    /**
     * The size of the second dimension.
     * @return The size of the second dimension.
     */
    public int getDimension2Size();

    /**
     * The number of non-zero elements in that row.
     * <p>
     * An element could be active and zero, if it was active on construction.
     *
     * @param row The index of the row.
     * @return The number of non-zero elements.
     */
    public int numActiveElements(int row);

    /**
     * Multiplies this Matrix by a {@link SGDVector} returning a vector of the appropriate size.
     * <p>
     * The input must have dimension equal to {@link Matrix#getDimension2Size()}.
     * @param input The input vector.
     * @return A new {@link SGDVector} of size {@link Matrix#getDimension1Size()}.
     */
    public SGDVector leftMultiply(SGDVector input);

    /**
     * Multiplies this Matrix by a {@link SGDVector} returning a vector of the appropriate size.
     * <p>
     * The input must have dimension equal to {@link Matrix#getDimension1Size()}.
     * @param input The input vector.
     * @return A new {@link SGDVector} of size {@link Matrix#getDimension2Size()}.
     */
    public SGDVector rightMultiply(SGDVector input);

    /**
     * Multiplies this Matrix by another {@link Matrix} returning a matrix of the appropriate size.
     * <p>
     * The input must have dimension 1 equal to {@link Matrix#getDimension2Size()}.
     * @param input The input matrix.
     * @return A new {@link Matrix} of size {@link Matrix#getDimension1Size()}, {@code input.getDimension2Size()}.
     */
    public Matrix matrixMultiply(Matrix input);

    /**
     * Multiplies this Matrix by another {@link Matrix} returning a matrix of the appropriate size.
     * <p>
     * Must obey the rules of matrix multiplication after the transposes are applied.
     * @param input The input matrix.
     * @param transposeThis Implicitly transposes this matrix just for the multiplication.
     * @param transposeOther Implicitly transposes other just for the multiplication.
     * @return A new {@link Matrix}.
     */
    public Matrix matrixMultiply(Matrix input, boolean transposeThis, boolean transposeOther);

    /**
     * Generates a {@link DenseVector} representing the sum of each column.
     * @return A new {@link DenseVector} of size {@link Matrix#getDimension2Size()}.
     */
    public DenseVector columnSum();

    /**
     * Generates a {@link DenseVector} representing the sum of each row.
     * @return A new {@link DenseVector} of size {@link Matrix#getDimension1Size()}.
     */
    public DenseVector rowSum();

    /**
     * Scales each row by the appropriate value in the {@link DenseVector}.
     * @param scalingCoefficients A {@link DenseVector} with size {@link Matrix#getDimension1Size()}.
     */
    public void rowScaleInPlace(DenseVector scalingCoefficients);

    /**
     * Reduces the rows of this matrix from left to right, producing a column vector.
     * <p>
     * The first argument to the reducer is the transformed element, the second is the state.
     * @param initialValue The initial value for the reduction.
     * @param op An operation to apply to each value.
     * @param reduction The reduction operation.
     * @return A vector containing the reduced values.
     */
    public DenseVector reduceRows(double initialValue, DoubleUnaryOperator op, DoubleBinaryOperator reduction);

    /**
     * An {@link SGDVector} view of the row.
     * <p>
     * This refers to the same values as the matrix, so updating this vector will update the matrix.
     * @param i The index of the row to extract.
     * @return An {@link SGDVector}.
     */
    public SGDVector getRow(int i);

    /**
     * Returns a copy of the specified column.
     * @param index The column index.
     * @return A copy of the column.
     */
    public SGDVector getColumn(int index);

    /**
     * Returns the index of the maximum element in each row.
     * @return The index of the row wise maximum elements.
     */
    int[] indexOfRowMax();

    /**
     * Aggregates an array of row vectors into a matrix.
     *
     * <p>The returned matrix will be a {@link DenseMatrix} if all the elements of {@code vectors}
     * are {@link DenseVector} instances. Otherwise it will be a {@link DenseSparseMatrix} and any
     * {@link DenseVector} instances will be copied into {@link SparseVector} instances.
     *
     * <p>If {@code requireCopy} is false then the matrix may be a view (or some rows may be a view)
     * on the original vectors, or it may be a copy.
     *
     * <p>If the vectors are not all of the same dimension then {@link IllegalArgumentException} is
     * thrown.
     * @param vectors The vectors to aggregate, must all be the same dimension.
     * @param length Aggregates the first {@code [0, length]} vectors, any vectors after length will be ignored.
     *               If {@code length} is greater than {@code vectors.length}, then
     *               an {@link IllegalArgumentException} is thrown.
     * @param requireCopy Requires the matrix to copy all the values.
     * @return A matrix aggregated from the supplied row vectors.
     */
    public static Matrix aggregate(SGDVector[] vectors, int length, boolean requireCopy) {
        // Validate args
        if (length < 1) {
            throw new IllegalArgumentException("Length must be positive, received " + length);
        }
        if (vectors.length == 0) {
            throw new IllegalArgumentException("Must supply a positive number of vectors.");
        }
        if (length > vectors.length) {
            throw new IllegalArgumentException("length " + length + ", vectors.length " + vectors.length);
        }
        int dim = vectors[0].size();
        boolean allDense = vectors[0] instanceof DenseVector;
        for (int i = 1; i < length; i++) {
            if (dim != vectors[i].size()) {
                throw new IllegalArgumentException("Vectors have different dimension, expected " + dim + ", found " + vectors[i].size());
            }
            allDense &= vectors[i] instanceof DenseVector;
        }
        if (allDense) {
            // Copy the arrays and return a dense matrix.
            double[][] arrays = new double[length][];
            for (int i = 0; i < arrays.length; i++) {
                if (requireCopy) {
                    arrays[i] = vectors[i].toArray();
                } else {
                    arrays[i] = ((DenseVector) vectors[i]).elements;
                }
            }
            return new DenseMatrix(arrays);
        } else {
            // All sparse, or a mixture of dense & sparse.
            SparseVector[] sparseVecs = new SparseVector[length];
            for (int i = 0; i < sparseVecs.length; i++) {
                if (vectors[i] instanceof DenseVector) {
                    // convert it
                    sparseVecs[i] = new SparseVector(vectors[i]);
                } else {
                    if (requireCopy) {
                        sparseVecs[i] = ((SparseVector) vectors[i]).copy();
                    } else {
                        sparseVecs[i] = ((SparseVector) vectors[i]);
                    }
                }
            }
            return new DenseSparseMatrix(sparseVecs);
        }
    }

    /**
     * Interface for matrix factorizations.
     */
    public interface Factorization {
        /**
         * First dimension of the factorized matrix.
         * @return First dimension size.
         */
        public int dim1();

        /**
         * Second dimension of the factorized matrix.
         * @return Second dimension size.
         */
        public int dim2();

        /**
         * Compute the matrix determinant of the factorized matrix.
         * @return The matrix determinant.
         */
        public double determinant();

        /**
         * Solves a system of linear equations A * b = y, where y is the input vector,
         * A is the matrix which produced this factorization, and b is the returned value.
         * @param vector The input vector y.
         * @return The vector b.
         */
        public SGDVector solve(SGDVector vector);

        /**
         * Solves the system A * X = Y, where Y is the input matrix, and A is the matrix which
         * produced this factorization.
         * @param matrix The input matrix Y.
         * @return The matrix X.
         */
        public Matrix solve(Matrix matrix);

        /**
         * Generates the inverse of the matrix with this factorization.
         * @return The matrix inverse.
         */
        default public Matrix inverse() {
            return solve(DenseSparseMatrix.createIdentity(dim2()));
        }
    }
}
