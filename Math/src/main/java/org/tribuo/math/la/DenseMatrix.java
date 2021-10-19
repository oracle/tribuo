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

import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

/**
 * A dense matrix, backed by a primitive array.
 */
public class DenseMatrix implements Matrix {
    private static final long serialVersionUID = 1L;

    private static final double DELTA = 1e-10;

    protected final double[][] values;
    protected final int dim1;
    protected final int dim2;

    private final int[] shape;

    private final int numElements;

    /**
     * Creates a dense matrix full of zeros.
     * @param dim1 The first dimension.
     * @param dim2 The second dimension.
     */
    public DenseMatrix(int dim1, int dim2) {
        this.values = new double[dim1][dim2];
        this.dim1 = dim1;
        this.dim2 = dim2;
        this.shape = new int[]{dim1,dim2};
        this.numElements = dim1*dim2;
    }

    /**
     * Copies the supplied matrix.
     * @param other The matrix to copy.
     */
    public DenseMatrix(DenseMatrix other) {
        this.values = new double[other.values.length][];
        for (int i = 0; i < values.length; i++) {
            this.values[i] = new double[other.values[i].length];
            for (int j = 0; j < values[i].length; j++) {
                this.values[i][j] = other.get(i,j);
            }
        }
        this.dim1 = other.dim1;
        this.dim2 = other.dim2;
        this.shape = new int[]{dim1,dim2};
        this.numElements = dim1*dim2;
    }

    /**
     * Copies the supplied matrix, densifying it if it's sparse.
     * @param other The matrix to copy.
     */
    public DenseMatrix(Matrix other) {
        this.dim1 = other.getDimension1Size();
        this.dim2 = other.getDimension2Size();
        this.values = new double[dim1][dim2];
        for (MatrixTuple t : other) {
            this.values[t.i][t.j] = t.value;
        }
        this.shape = new int[]{dim1,dim2};
        this.numElements = dim1*dim2;
    }

    /**
     * Creates a DenseMatrix without defensive copying.
     * @param values The values of the matrix.
     */
    DenseMatrix(double[][] values) {
        this.values = values;
        this.dim1 = values.length;
        this.dim2 = values[0].length;
        this.shape = new int[]{dim1,dim2};
        this.numElements = dim1*dim2;
    }

    /**
     * Defensively copies the values before construction.
     * <p>
     * Throws IllegalArgumentException if the supplied values are a ragged array.
     * @param values The values of this dense matrix.
     * @return A new dense matrix.
     */
    public static DenseMatrix createDenseMatrix(double[][] values) {
        double[][] newValues = new double[values.length][];
        int sizeCounter = -1;
        for (int i = 0; i < newValues.length; i++) {
            if (sizeCounter == -1) {
                sizeCounter = values[i].length;
            }
            if (sizeCounter != values[i].length) {
                throw new IllegalArgumentException("DenseMatrix must not be ragged. Expected dim2 = " + sizeCounter + ", but found " + values[i].length + " at index " + i);
            }
            newValues[i] = Arrays.copyOf(values[i],values[i].length);
        }
        return new DenseMatrix(newValues);
    }

    @Override
    public int[] getShape() {
        return shape;
    }

    @Override
    public Tensor reshape(int[] newShape) {
        int sum = Tensor.shapeSum(newShape);
        if (sum != numElements) {
            throw new IllegalArgumentException("Invalid shape " + Arrays.toString(newShape) + ", expected something with " + numElements + " elements.");
        }

        if (newShape.length == 2) {
            DenseMatrix matrix = new DenseMatrix(newShape[0],newShape[1]);

            for (int a = 0; a < numElements; a++) {
                int oldI = a % dim1;
                int oldJ = a % dim2;
                int i = a % newShape[0];
                int j = a / newShape[0];
                matrix.set(i,j,get(oldI,oldJ));
            }

            return matrix;
        } else if (newShape.length == 1) {
            DenseVector vector = new DenseVector(numElements);
            int a = 0;
            for (int i = 0; i < dim1; i++) {
                for (int j = 0; j < dim2; j++) {
                    vector.set(a,get(i,j));
                    a++;
                }
            }
            return vector;
        } else {
            throw new IllegalArgumentException("Only supports 1 or 2 dimensional tensors.");
        }
    }

    /**
     * Copies the matrix.
     * @return A deep copy of the matrix.
     */
    @Override
    public DenseMatrix copy() {
        return new DenseMatrix(this);
    }

    @Override
    public double get(int i, int j) {
        return values[i][j];
    }

    /**
     * Constructs a dense vector by gathering values across dimension 1.
     * @param elements The indices to gather.
     * @return A dense vector.
     */
    public DenseVector gatherAcrossDim1(int[] elements) {
        if (elements.length != dim2) {
            throw new IllegalArgumentException("Invalid number of elements to gather, must select one per value of dim2");
        }
        double[] outputValues = new double[dim2];

        for (int i = 0; i < elements.length; i++) {
            outputValues[i] = values[elements[i]][i];
        }

        return new DenseVector(outputValues);
    }

    /**
     * Constructs a dense vector by gathering values across dimension 2.
     * @param elements The indices to gather.
     * @return A dense vector.
     */
    public DenseVector gatherAcrossDim2(int[] elements) {
        if (elements.length != dim1) {
            throw new IllegalArgumentException("Invalid number of elements to gather, must select one per value of dim1");
        }
        double[] outputValues = new double[dim1];

        for (int i = 0; i < elements.length; i++) {
            outputValues[i] = values[i][elements[i]];
        }

        return new DenseVector(outputValues);
    }

    /**
     * Returns a transposed copy of this matrix.
     * @return A transposed copy.
     */
    public DenseMatrix transpose() {
        double[][] newValues = new double[dim2][dim1];

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                newValues[j][i] = get(i,j);
            }
        }

        return new DenseMatrix(newValues);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DenseMatrix)) return false;
        DenseMatrix that = (DenseMatrix) o;
        if ((dim1 == that.dim1) && (dim2 == that.dim2) && (numElements == that.numElements) && Arrays.equals(getShape(),that.getShape())) {
            for (int i = 0; i < dim1; i++) {
                for (int j = 0; j < dim2; j++) {
                    if (Math.abs(get(i,j) - that.get(i,j)) > DELTA) {
                        return false;
                    }
                }
            }
            return true;
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(dim1, dim2, numElements);
        result = 31 * result + Arrays.hashCode(values);
        result = 31 * result + Arrays.hashCode(getShape());
        return result;
    }

    @Override
    public void set(int i, int j, double value) {
        values[i][j] = value;
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
            if (input instanceof DenseVector) {
                // If it's dense we can use loops
                for (int i = 0; i < dim1; i++) {
                    for (int j = 0; j < dim2; j++) {
                        output[i] += get(i,j) * input.get(j);
                    }
                }
            } else {
                // If it's sparse we iterate the tuples
                for (VectorTuple tuple : input) {
                    for (int i = 0; i < output.length; i++) {
                        output[i] += values[i][tuple.index] * tuple.value;
                    }
                }
            }
            return new DenseVector(output);
        } else {
            throw new IllegalArgumentException("input.size() != dim2, input.size() = " + input.size() + ", dim1,dim2 = " + dim1+","+dim2);
        }
    }

    @Override
    public DenseVector rightMultiply(SGDVector input) {
        if (input.size() == dim1) {
            double[] output = new double[dim2];
            if (input instanceof DenseVector) {
                // If it's dense we can use loops
                for (int i = 0; i < dim1; i++) {
                    double curValue = input.get(i);
                    for (int j = 0; j < dim2; j++) {
                        output[j] += get(i,j) * curValue;
                    }
                }
            } else {
                // If it's sparse we iterate the tuples
                for (VectorTuple tuple : input) {
                    for (int i = 0; i < output.length; i++) {
                        output[i] += values[tuple.index][i] * tuple.value;
                    }
                }
            }
            return new DenseVector(output);
        } else {
            throw new IllegalArgumentException("input.size() != dim1");
        }
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
            throw new IllegalArgumentException("Invalid matrix dimensions, this.shape=" + Arrays.toString(shape) + ", other.shape = " + Arrays.toString(other.getShape()));
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
            throw new IllegalArgumentException("Invalid matrix dimensions, this.shape=" + Arrays.toString(shape) + ", other.shape = " + Arrays.toString(other.getShape()));
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
            throw new IllegalArgumentException("Invalid matrix dimensions, this.shape=" + Arrays.toString(shape) + ", other.shape = " + Arrays.toString(other.getShape()));
        }
    }

    private double columnRowDot(int rowIndex, int otherColIndex, Matrix other) {
        double sum = 0.0;
        for (int i = 0; i < dim2; i++) {
            sum += get(rowIndex,i) * other.get(i,otherColIndex);
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
        for (int i = 0; i < dim2; i++) {
            sum += get(rowIndex,i) * other.get(otherRowIndex,i);
        }
        return sum;
    }

    @Override
    public DenseVector rowSum() {
        double[] rowSum = new double[dim1];
        for (int i = 0; i < dim1; i++) {
            double tmp = 0.0;
            for (int j = 0; j < dim2; j++) {
                tmp += values[i][j];
            }
            rowSum[i] = tmp;
        }
        return new DenseVector(rowSum);
    }

    @Override
    public void rowScaleInPlace(DenseVector scalingCoefficients) {
        for (int i = 0; i < dim1; i++) {
            double scalar = scalingCoefficients.get(i);
            for (int j = 0; j < dim2; j++) {
                values[i][j] *= scalar;
            }
        }
    }

    @Override
    public void add(int i, int j, double value) {
        values[i][j] += value;
    }

    /**
     * Adds the specified value to the specified elements across dimension 1.
     * @param indices The indices to update.
     * @param value The value to add.
     */
    public void addAcrossDim1(int[] indices, double value) {
        if (indices.length != dim2) {
            throw new IllegalArgumentException("Invalid number of elements to add, must select one per value of dim2");
        }
        for (int i = 0; i < indices.length; i++) {
            values[indices[i]][i] += value;
        }
    }

    /**
     * Adds the specified value to the specified elements across dimension 2.
     * @param indices The indices to update.
     * @param value The value to add.
     */
    public void addAcrossDim2(int[] indices, double value) {
        if (indices.length != dim1) {
            throw new IllegalArgumentException("Invalid number of elements to indices, must select one per value of dim1");
        }
        for (int i = 0; i < indices.length; i++) {
            values[i][indices[i]] += value;
        }
    }

    @Override
    public void intersectAndAddInPlace(Tensor other, DoubleUnaryOperator f) {
        if (other instanceof Matrix) {
            Matrix otherMat = (Matrix) other;
            if ((dim1 == otherMat.getDimension1Size()) && (dim2 == otherMat.getDimension2Size())) {
                if (otherMat instanceof DenseMatrix) {
                    // Get is efficient on DenseMatrix
                    for (int i = 0; i < dim1; i++) {
                        for (int j = 0; j < dim2; j++) {
                            values[i][j] += f.applyAsDouble(otherMat.get(i,j));
                        }
                    }
                } else {
                    // Fall back to tuple based iteration
                    for (MatrixTuple tuple : otherMat) {
                        values[tuple.i][tuple.j] += f.applyAsDouble(tuple.value);
                    }
                }
            } else {
                throw new IllegalArgumentException("Matrices are not the same size, this("+dim1+","+dim2+"), other("+otherMat.getDimension1Size()+","+otherMat.getDimension2Size()+")");
            }
        } else {
            throw new IllegalArgumentException("Adding a non-Matrix to a Matrix");
        }
    }

    @Override
    public void hadamardProductInPlace(Tensor other, DoubleUnaryOperator f) {
        if (other instanceof Matrix) {
            Matrix otherMat = (Matrix) other;
            if ((dim1 == otherMat.getDimension1Size()) && (dim2 == otherMat.getDimension2Size())) {
                if (otherMat instanceof DenseMatrix) {
                    // Get is efficient on DenseMatrix
                    for (int i = 0; i < dim1; i++) {
                        for (int j = 0; j < dim2; j++) {
                            values[i][j] *= f.applyAsDouble(otherMat.get(i,j));
                        }
                    }
                } else {
                    // Fall back to tuple based iteration
                    for (MatrixTuple tuple : otherMat) {
                        values[tuple.i][tuple.j] *= f.applyAsDouble(tuple.value);
                    }
                }
            } else {
                throw new IllegalArgumentException("Matrices are not the same size, this("+dim1+","+dim2+"), other("+otherMat.getDimension1Size()+","+otherMat.getDimension2Size()+")");
            }
        } else {
            throw new IllegalArgumentException("Adding a non-Matrix to a Matrix");
        }
    }

    @Override
    public void foreachInPlace(DoubleUnaryOperator f) {
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < dim2; j++) {
                values[i][j] = f.applyAsDouble(values[i][j]);
            }
        }
    }

    /**
     * Broadcasts the input vector and adds it to each row/column of the matrix.
     * @param input The input vector.
     * @param broadcastOverDim1 If true broadcasts over the first dimension, else broadcasts over the second.
     */
    public void broadcastIntersectAndAddInPlace(SGDVector input, boolean broadcastOverDim1) {
        if (input instanceof DenseVector) {
            if (broadcastOverDim1) {
                if (input.size() == dim2) {
                    for (int i = 0; i < dim1; i++) {
                        for (int j = 0; j < dim2; j++) {
                            values[i][j] += input.get(j);
                        }
                    }
                } else {
                    throw new IllegalArgumentException("Input vector must have dimension equal to dim 2, input.size() = " + input.size() + ", dim2 = " + dim2);
                }
            } else {
                if (input.size() == dim1) {
                    for (int i = 0; i < dim1; i++) {
                        double ith = input.get(i);
                        for (int j = 0; j < dim2; j++) {
                            values[i][j] += ith;
                        }
                    }
                } else {
                    throw new IllegalArgumentException("Input vector must have dimension equal to dim 1, input.size() = " + input.size() + ", dim1 = " + dim1);
                }
            }
        } else if (input instanceof SparseVector) {
            if (broadcastOverDim1) {
                if (input.size() == dim2) {
                    for (int i = 0; i < dim1; i++) {
                        for (VectorTuple v : input) {
                            values[i][v.index] += v.value;
                        }
                    }
                } else {
                    throw new IllegalArgumentException("Input vector must have dimension equal to dim 2, input.size() = " + input.size() + ", dim2 = " + dim2);
                }
            } else {
                if (input.size() == dim1) {
                    for (VectorTuple v : input) {
                        for (int j = 0; j < dim2; j++) {
                            values[v.index][j] += v.value;
                        }
                    }
                } else {
                    throw new IllegalArgumentException("Input vector must have dimension equal to dim 1, input.size() = " + input.size() + ", dim1 = " + dim1);
                }
            }

        } else {
            throw new IllegalArgumentException("Input vector was neither dense nor sparse.");
        }
    }

    @Override
    public int numActiveElements(int row) {
        return dim2;
    }

    @Override
    public DenseVector getRow(int i) {
        return new DenseVector(values[i]);
    }

    /**
     * Returns a copy of the specified column.
     * @param index The column index.
     * @return A copy of the column.
     */
    public DenseVector getColumn(int index) {
        double[] output = new double[dim1];
        for (int i = 0; i < dim1; i++) {
            output[i] = values[i][index];
        }
        return new DenseVector(output);
    }

    /**
     * Calculates the sum of the specified row.
     * @param rowIndex The index of the row to sum.
     * @return The row sum.
     */
    public double rowSum(int rowIndex) {
        double[] row = values[rowIndex];
        double sum = 0d;
        for (int i = 0; i < row.length; i++) {
            sum += row[i];
        }
        return sum;
    }

    /**
     * Calculates the sum of the specified column.
     * @param columnIndex The index of the column to sum.
     * @return The column sum.
     */
    public double columnSum(int columnIndex) {
        double sum = 0d;
        for (int i = 0; i < dim1; i++) {
            sum += values[i][columnIndex];
        }
        return sum;
    }

    @Override
    public double twoNorm() {
        double output = 0.0;
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                double value = get(i,j);
                output += value * value;
            }
        }
        return Math.sqrt(output);
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("DenseMatrix(dim1=");
        buffer.append(dim1);
        buffer.append(",dim2=");
        buffer.append(dim2);
        buffer.append(",values=\n");
        for (int i = 0; i < dim1; i++) {
            buffer.append("\trow ");
            buffer.append(i);
            buffer.append(" [");
            for (int j = 0; j < dim2; j++) {
                if (values[i][j] < 0.0) {
                    buffer.append(String.format("%.15f", values[i][j]));
                } else {
                    buffer.append(String.format(" %.15f", values[i][j]));
                }
                buffer.append(",");
            }
            buffer.deleteCharAt(buffer.length()-1);
            buffer.append("];\n");
        }
        buffer.append(")");

        return buffer.toString();
    }

    @Override
    public MatrixIterator iterator() {
        return new DenseMatrixIterator(this);
    }

    /**
     * Normalizes each row using the supplied normalizer in place.
     * @param normalizer The vector normalizer to use.
     */
    public void normalizeRows(VectorNormalizer normalizer) {
        for (int i = 0; i < dim1; i++) {
            normalizer.normalizeInPlace(values[i]);
        }
    }

    /**
     * Returns the dense vector containing each column sum.
     * @return The column sums.
     */
    public DenseVector columnSum() {
        double[] columnSum = new double[dim2];
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                columnSum[j] += values[i][j];
            }
        }
        return new DenseVector(columnSum);
    }

    private class DenseMatrixIterator implements MatrixIterator {
        private final DenseMatrix matrix;
        private final MatrixTuple tuple;
        private int i;
        private int j;

        public DenseMatrixIterator(DenseMatrix matrix) {
            this.matrix = matrix;
            this.tuple = new MatrixTuple();
            this.i = 0;
            this.j = 0;
        }

        @Override
        public MatrixTuple getReference() {
            return tuple;
        }

        @Override
        public boolean hasNext() {
            return (i < matrix.dim1) && (j < matrix.dim2);
        }

        @Override
        public MatrixTuple next() {
            if (!hasNext()) {
                throw new NoSuchElementException("Off the end of the iterator.");
            }
            tuple.i = i;
            tuple.j = j;
            tuple.value = matrix.values[i][j];
            if (j < dim2-1) {
                j++;
            } else {
                //Reached end of current vector, get next one
                i++;
                j = 0;
            }
            return tuple;
        }
    }

}
