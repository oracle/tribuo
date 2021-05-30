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

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

/**
 * A matrix which is dense in the first dimension and sparse in the second.
 * <p>
 * Backed by an array of {@link SparseVector}.
 */
public class DenseSparseMatrix implements Matrix {
    private static final long serialVersionUID = 1L;

    private final SparseVector[] values;
    private final int dim1;
    private final int dim2;
    private final int[] shape;

    DenseSparseMatrix(SparseVector[] values) {
        this.values = values;
        this.dim1 = values.length;
        this.dim2 = values[0].size();
        this.shape = new int[]{dim1,dim2};
    }

    /***
     * Constructs a DenseSparseMatrix out of the supplied sparse vector list.
     * <p>
     * Throws IllegalArgumentException if the list forms a ragged matrix.
     * <p>
     * Does not copy the vectors, it directly wraps them.
     * @param values The SparseVectors.
     */
    public DenseSparseMatrix(List<SparseVector> values) {
        this.values = new SparseVector[values.size()];
        this.dim1 = values.size();
        this.dim2 = values.get(0).size();
        this.shape = new int[]{dim1,dim2};
        for (int i = 0; i < values.size(); i++) {
            if (values.get(i).size() != dim2) {
                throw new IllegalArgumentException("Unexpected size, found " + values.get(i).size() + ", expected " + dim2);
            }
            this.values[i] = values.get(i);
            if (values.get(i).size() != this.dim2) {
                throw new IllegalArgumentException("Vectors are not the same dimension, expected " + dim2 + ", found " + values.get(i).size() + " at index " + i);
            }
        }
    }

    /**
     * Creates a new DenseSparseMatrix by deep copying the supplied DenseSparseMatrix.
     * @param other The matrix to copy.
     */
    public DenseSparseMatrix(DenseSparseMatrix other) {
        this.dim1 = other.dim1;
        this.dim2 = other.dim2;
        this.values = new SparseVector[other.values.length];
        this.shape = new int[]{dim1,dim2};
        for (int i = 0; i < values.length; i++) {
            values[i] = other.values[i].copy();
        }
    }

    /**
     * Creates a DenseSparseMatrix with no values or indices.
     * <p>
     * Used as a placeholder in the gradient system.
     * @param dim1 The first dimension.
     * @param dim2 The second dimension.
     */
    public DenseSparseMatrix(int dim1, int dim2) {
        this.dim1 = dim1;
        this.dim2 = dim2;
        this.values = new SparseVector[dim1];
        this.shape = new int[]{dim1,dim2};
        SparseVector emptyVector = new SparseVector(dim2);
        for (int i = 0; i < values.length; i++) {
            values[i] = emptyVector;
        }
    }

    /**
     * Defensively copies the values.
     * @param values The sparse vectors to use.
     * @return A DenseSparseMatrix containing the supplied vectors.
     */
    public static DenseSparseMatrix createFromSparseVectors(SparseVector[] values) {
        SparseVector[] newValues = new SparseVector[values.length];
        for (int i = 0; i < values.length; i++) {
            newValues[i] = values[i].copy();
        }
        return new DenseSparseMatrix(newValues);
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
    public DenseSparseMatrix copy() {
        SparseVector[] copies = new SparseVector[values.length];

        for (int i = 0; i < values.length; i++) {
            copies[i] = values[i].copy();
        }

        return new DenseSparseMatrix(copies);
    }

    @Override
    public double get(int i, int j) {
        return values[i].get(j);
    }

    @Override
    public void set(int i, int j, double value) {
        values[i].set(j,value);
    }

    @Override
    public int getDimension1Size() {
        return dim1;
    }

    @Override
    public int getDimension2Size() {
        return dim2;
    }

    @Override
    public DenseVector leftMultiply(SGDVector input) {
        if (input.size() == dim2) {
            double[] output = new double[dim1];

            for (int i = 0; i < output.length; i++) {
                output[i] = values[i].dot(input);
            }

            return new DenseVector(output);
        } else {
            throw new IllegalArgumentException("input.size() != dim2");
        }
    }

    /**
     * rightMultiply is very inefficient on DenseSparseMatrix due to the storage format.
     * @param input The input vector.
     * @return A*input.
     */
    @Override
    public DenseVector rightMultiply(SGDVector input) {
        if (input.size() == dim1) {
            double[] output = new double[dim2];

            for (int j = 0; j < values.length; j++) {
                for (int i = 0; i < output.length; i++) {
                    output[i] = values[j].get(i) * input.get(i);
                }
            }

            return new DenseVector(output);
        } else {
            throw new IllegalArgumentException("input.size() != dim1");
        }
    }

    @Override
    public void add(int i, int j, double value) {
        values[i].add(j,value);
    }

    /**
     * Only implemented for {@link DenseMatrix}.
     * @param other The other {@link Tensor}.
     * @param f A function to apply.
     */
    @Override
    public void intersectAndAddInPlace(Tensor other, DoubleUnaryOperator f) {
        if (other instanceof Matrix) {
            Matrix otherMat = (Matrix) other;
            if ((dim1 == otherMat.getDimension1Size()) && (dim2 == otherMat.getDimension2Size())) {
                if (otherMat instanceof DenseMatrix) {
                    DenseMatrix otherDenseMat = (DenseMatrix) other;
                    for (int i = 0; i < dim1; i++) {
                        values[i].intersectAndAddInPlace(otherDenseMat.getRow(i),f);
                    }
                } else {
                    throw new UnsupportedOperationException("Not implemented intersectAndAddInPlace in DenseSparseMatrix for types other than DenseMatrix");
                }
            } else {
                throw new IllegalArgumentException("Matrices are not the same size, this("+dim1+","+dim2+"), other("+otherMat.getDimension1Size()+","+otherMat.getDimension2Size()+")");
            }
        } else {
            throw new IllegalArgumentException("Adding a non-Matrix to a Matrix");
        }
    }

    /**
     * Only implemented for {@link DenseMatrix}.
     * @param other The other {@link Tensor}.
     * @param f A function to apply.
     */
    @Override
    public void hadamardProductInPlace(Tensor other, DoubleUnaryOperator f) {
        if (other instanceof Matrix) {
            Matrix otherMat = (Matrix) other;
            if ((dim1 == otherMat.getDimension1Size()) && (dim2 == otherMat.getDimension2Size())) {
                if (otherMat instanceof DenseMatrix) {
                    DenseMatrix otherDenseMat = (DenseMatrix) other;
                    for (int i = 0; i < dim1; i++) {
                        values[i].hadamardProductInPlace(otherDenseMat.getRow(i),f);
                    }
                } else {
                    throw new UnsupportedOperationException("Not implemented hadamardProductInPlace in DenseSparseMatrix for types other than DenseMatrix");
                }
            } else {
                throw new IllegalArgumentException("Matrices are not the same size, this("+dim1+","+dim2+"), other("+otherMat.getDimension1Size()+","+otherMat.getDimension2Size()+")");
            }
        } else {
            throw new IllegalArgumentException("Scaling a Matrix by a non-Matrix");
        }
    }

    @Override
    public void foreachInPlace(DoubleUnaryOperator f) {
        for (int i = 0; i < values.length; i++) {
            values[i].foreachInPlace(f);
        }
    }

    @Override
    public int numActiveElements(int row) {
        return values[row].numActiveElements();
    }

    @Override
    public SparseVector getRow(int i) {
        return values[i];
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof Matrix) {
            Iterator<MatrixTuple> ourItr = iterator();
            Iterator<MatrixTuple> otherItr = ((Matrix)other).iterator();
            MatrixTuple ourTuple;
            MatrixTuple otherTuple;

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
        int result = Objects.hash(dim1, dim2);
        result = 31 * result + Arrays.hashCode(values);
        return result;
    }

    @Override
    public double twoNorm() {
        double output = 0.0;
        for (int i = 0; i < dim1; i++) {
            double value = values[i].twoNorm();
            output += value * value;
        }
        return Math.sqrt(output);
    }

    @Override
    public DenseMatrix matrixMultiply(Matrix other) {
        if (dim2 == other.getDimension1Size()) {
            if (other instanceof DenseMatrix) {
                DenseMatrix otherDense = (DenseMatrix) other;
                double[][] output = new double[dim1][otherDense.dim2];

                for (int i = 0; i < dim1; i++) {
                    for (int j = 0; j < otherDense.dim2; j++) {
                        output[i][j] = columnRowDot(i,j,otherDense);
                    }
                }

                return new DenseMatrix(output);
            } else if (other instanceof DenseSparseMatrix) {
                DenseSparseMatrix otherSparse = (DenseSparseMatrix) other;
                int otherDim2 = otherSparse.getDimension2Size();
                double[][] output = new double[dim1][otherDim2];

                for (int i = 0; i < dim1; i++) {
                    for (int j = 0; j < otherDim2; j++) {
                        output[i][j] = columnRowDot(i,j,otherSparse);
                    }
                }

                return new DenseMatrix(output);
            } else {
                throw new IllegalArgumentException("Unknown matrix type " + other.getClass().getName());
            }
        } else {
            throw new IllegalArgumentException("Invalid matrix dimensions, this.shape=" + Arrays.toString(shape) + ", other.shape = " + Arrays.toString(other.getShape()));
        }
    }

    @Override
    public DenseMatrix matrixMultiply(Matrix other, boolean transposeThis, boolean transposeOther) {
        if (transposeThis && transposeOther) {
            return matrixMultiplyTransposeBoth(other);
        } else if (transposeThis) {
            return matrixMultiplyTransposeThis(other);
        } else if (transposeOther) {
            return matrixMultiplyTransposeOther(other);
        } else {
            return matrixMultiply(other);
        }
    }

    private DenseMatrix matrixMultiplyTransposeBoth(Matrix other) {
        if (dim1 == other.getDimension2Size()) {
            if (other instanceof DenseMatrix) {
                DenseMatrix otherDense = (DenseMatrix) other;
                double[][] output = new double[dim2][otherDense.dim1];

                for (int i = 0; i < dim2; i++) {
                    for (int j = 0; j < otherDense.dim1; j++) {
                        output[i][j] = rowColumnDot(i,j,otherDense);
                    }
                }

                return new DenseMatrix(output);
            } else if (other instanceof DenseSparseMatrix) {
                DenseSparseMatrix otherSparse = (DenseSparseMatrix) other;
                int otherDim1 = otherSparse.getDimension1Size();
                double[][] output = new double[dim2][otherDim1];

                for (int i = 0; i < dim2; i++) {
                    for (int j = 0; j < otherDim1; j++) {
                        output[i][j] = rowColumnDot(i,j,otherSparse);
                    }
                }

                return new DenseMatrix(output);
            } else {
                throw new IllegalArgumentException("Unknown matrix type " + other.getClass().getName());
            }
        } else {
            throw new IllegalArgumentException("Invalid matrix dimensions, dim1 = " + dim1 + ", other.dim2 = " + other.getDimension2Size());
        }
    }

    private DenseMatrix matrixMultiplyTransposeThis(Matrix other) {
        if (dim1 == other.getDimension1Size()) {
            if (other instanceof DenseMatrix) {
                DenseMatrix otherDense = (DenseMatrix) other;
                double[][] output = new double[dim2][otherDense.dim2];

                for (int i = 0; i < dim2; i++) {
                    for (int j = 0; j < otherDense.dim2; j++) {
                        output[i][j] = columnColumnDot(i,j,otherDense);
                    }
                }

                return new DenseMatrix(output);
            } else if (other instanceof DenseSparseMatrix) {
                DenseSparseMatrix otherSparse = (DenseSparseMatrix) other;
                int otherDim2 = otherSparse.getDimension2Size();
                double[][] output = new double[dim2][otherDim2];

                for (int i = 0; i < dim2; i++) {
                    for (int j = 0; j < otherDim2; j++) {
                        output[i][j] = columnColumnDot(i,j,otherSparse);
                    }
                }

                return new DenseMatrix(output);
            } else {
                throw new IllegalArgumentException("Unknown matrix type " + other.getClass().getName());
            }
        } else {
            throw new IllegalArgumentException("Invalid matrix dimensions, dim1 = " + dim1 + ", other.dim1 = " + other.getDimension1Size());
        }
    }

    private DenseMatrix matrixMultiplyTransposeOther(Matrix other) {
        if (dim2 == other.getDimension2Size()) {
            if (other instanceof DenseMatrix) {
                DenseMatrix otherDense = (DenseMatrix) other;
                double[][] output = new double[dim1][otherDense.dim1];

                for (int i = 0; i < dim1; i++) {
                    for (int j = 0; j < otherDense.dim1; j++) {
                        output[i][j] = rowRowDot(i,j,otherDense);
                    }
                }

                return new DenseMatrix(output);
            } else if (other instanceof DenseSparseMatrix) {
                DenseSparseMatrix otherSparse = (DenseSparseMatrix) other;
                int otherDim1 = otherSparse.getDimension1Size();
                double[][] output = new double[dim1][otherDim1];

                for (int i = 0; i < dim1; i++) {
                    for (int j = 0; j < otherDim1; j++) {
                        output[i][j] = rowRowDot(i,j,otherSparse);
                    }
                }

                return new DenseMatrix(output);
            } else {
                throw new IllegalArgumentException("Unknown matrix type " + other.getClass().getName());
            }
        } else {
            throw new IllegalArgumentException("Invalid matrix dimensions, dim2 = " + dim2 + ", other.dim2 = " + other.getDimension2Size());
        }
    }

    private double columnRowDot(int rowIndex, int otherColIndex, Matrix other) {
        double sum = 0.0;
        for (VectorTuple tuple : values[rowIndex]) {
            sum += tuple.value * other.get(tuple.index,otherColIndex);
        }
        return sum;
    }

    private double rowColumnDot(int colIndex, int otherRowIndex, Matrix other) {
        double sum = 0.0;
        for (int i = 0; i < dim1; i++) {
            sum += get(i,colIndex) * other.get(otherRowIndex,i);
        }
        return sum;
    }

    private double columnColumnDot(int colIndex, int otherColIndex, Matrix other) {
        double sum = 0.0;
        for (int i = 0; i < dim1; i++) {
            sum += get(i,colIndex) * other.get(i,otherColIndex);
        }
        return sum;
    }

    private double rowRowDot(int rowIndex, int otherRowIndex, Matrix other) {
        double sum = 0.0;
        for (VectorTuple tuple : values[rowIndex]) {
            sum += tuple.value * other.get(otherRowIndex,tuple.index);
        }
        return sum;
    }

    @Override
    public DenseVector rowSum() {
        double[] rowSum = new double[dim1];
        for (int i = 0; i < dim1; i++) {
            rowSum[i] = values[i].sum();
        }
        return new DenseVector(rowSum);
    }

    @Override
    public void rowScaleInPlace(DenseVector scalingCoefficients) {
        for (int i = 0; i < dim1; i++) {
            values[i].scaleInPlace(scalingCoefficients.get(i));
        }
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("DenseSparseMatrix(\n");
        for (int i = 0; i < values.length; i++) {
            buffer.append("\t");
            buffer.append(values[i].toString());
            buffer.append(";\n");
        }
        buffer.append(")");

        return buffer.toString();
    }

    @Override
    public MatrixIterator iterator() {
        return new DenseSparseMatrixIterator(this);
    }

    private static class DenseSparseMatrixIterator implements MatrixIterator {
        private final DenseSparseMatrix matrix;
        private final MatrixTuple tuple;
        private int i;
        private Iterator<VectorTuple> itr;
        private VectorTuple vecTuple;

        public DenseSparseMatrixIterator(DenseSparseMatrix matrix) {
            this.matrix = matrix;
            this.tuple = new MatrixTuple();
            this.i = 0;
            this.itr = matrix.values[0].iterator();
        }

        @Override
        public String toString() {
            return "DenseSparseMatrixIterator(position="+i+",tuple="+ tuple.toString()+")";
        }

        @Override
        public MatrixTuple getReference() {
            return tuple;
        }

        @Override
        public boolean hasNext() {
            if (itr.hasNext()) {
                return true;
            } else {
                while ((i < matrix.dim1) && (!itr.hasNext())) {
                    i++;
                    if (i < matrix.dim1) {
                        itr = matrix.values[i].iterator();
                    }
                }
            }
            return (i < matrix.dim1) && itr.hasNext();
        }

        @Override
        public MatrixTuple next() {
            vecTuple = itr.next();
            tuple.i = i;
            tuple.j = vecTuple.index;
            tuple.value = vecTuple.value;
            return tuple;
        }
    }
}
