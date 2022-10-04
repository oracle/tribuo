/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.SortUtil;
import org.tribuo.math.protos.DenseTensorProto;
import org.tribuo.math.protos.TensorProto;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.util.Util;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Optional;
import java.util.function.DoubleUnaryOperator;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * A dense matrix, backed by a primitive array.
 */
public class DenseMatrix implements Matrix {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(DenseMatrix.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Tolerance for non-zero diagonal values in the factorizations.
     */
    public static final double FACTORIZATION_TOLERANCE = 1e-14;

    private static final double DELTA = 1e-10;

    /**
     * The value array.
     */
    protected final double[][] values;
    /**
     * The number of rows.
     */
    protected final int dim1;
    /**
     * The number of columns.
     */
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
        if (other instanceof DenseMatrix) {
            for (int i = 0; i < dim1; i++) {
                for (int j = 0; j < dim2; j++) {
                    this.values[i][j] = other.get(i,j);
                }
            }
        } else {
            for (MatrixTuple t : other) {
                this.values[t.i][t.j] = t.value;
            }
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

    /**
     * Constructs a new DenseMatrix copying the values from the supplied vectors.
     * <p>
     * Throws {@link IllegalArgumentException} if the supplied vectors are ragged (i.e., are not all the same size).
     * @param vectors The vectors to coalesce.
     * @return A new dense matrix.
     */
    public static DenseMatrix createDenseMatrix(SGDVector[] vectors) {
        if (vectors == null || vectors.length == 0) {
            throw new IllegalArgumentException("Invalid vector array.");
        }
        double[][] newValues = new double[vectors.length][];

        int size = vectors[0].size();
        for (int i = 0; i < vectors.length; i++) {
            if (vectors[i].size() != size) {
                throw new IllegalArgumentException("Expected size " + size + " but found size " + vectors[i].size() + " at index " + i);
            }
            newValues[i] = vectors[i].toArray();
        }

        return new DenseMatrix(newValues);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static DenseMatrix deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        DenseTensorProto proto = message.unpack(DenseTensorProto.class);
        return unpackProto(proto);
    }

    /**
     * Unpacks a {@link DenseTensorProto} into a {@link DenseMatrix}.
     * @param proto The proto to unpack.
     * @return The dense matrix.
     */
    protected static DenseMatrix unpackProto(DenseTensorProto proto) {
        List<Integer> shapeList = proto.getDimensionsList();
        int[] shape = Util.toPrimitiveInt(shapeList);
        if (shape.length != 2) {
            throw new IllegalArgumentException("Invalid proto, expected a matrix, found shape " + Arrays.toString(shape));
        }
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 1) {
                throw new IllegalArgumentException("Invalid proto, shape must be positive, found " + shape[i] + " at position " + i);
            }
        }
        int numElements = Util.product(shape);
        DoubleBuffer buffer = proto.getValues().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        if (buffer.remaining() != numElements) {
            throw new IllegalArgumentException("Invalid proto, claimed " + numElements + ", but only had " + buffer.remaining() + " values");
        }
        double[][] values = new double[shape[0]][shape[1]];
        for (int i = 0; i < values.length; i++) {
            buffer.get(values[i]);
        }
        return new DenseMatrix(values);
    }

    @Override
    public TensorProto serialize() {
        TensorProto.Builder builder = TensorProto.newBuilder();

        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(DenseMatrix.class.getName());

        DenseTensorProto.Builder dataBuilder = DenseTensorProto.newBuilder();
        dataBuilder.addAllDimensions(Arrays.stream(shape).boxed().collect(Collectors.toList()));
        ByteBuffer buffer = ByteBuffer.allocate(numElements * 8).order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer doubleBuffer = buffer.asDoubleBuffer();
        for (int i = 0; i < values.length; i ++) {
            doubleBuffer.put(values[i]);
        }
        doubleBuffer.rewind();
        dataBuilder.setValues(ByteString.copyFrom(buffer));
        builder.setSerializedData(Any.pack(dataBuilder.build()));

        return builder.build();
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
            outputValues[i] = get(elements[i],i);
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
            outputValues[i] = get(i,elements[i]);
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
        result = 31 * result + Arrays.deepHashCode(values);
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
                        output[i] += get(i,tuple.index) * tuple.value;
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
                        output[i] += get(tuple.index,i) * tuple.value;
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
                tmp += get(i,j);
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
        if (i < 0 || i > dim1) {
            throw new IllegalArgumentException("Invalid row index, must be [0,"+dim1+"), received " + i);
        }
        return new DenseVector(values[i]);
    }

    /**
     * Returns a copy of the specified column.
     * @param index The column index.
     * @return A copy of the column.
     */
    @Override
    public DenseVector getColumn(int index) {
        if (index < 0 || index > dim2) {
            throw new IllegalArgumentException("Invalid column index, must be [0,"+dim2+"), received " + index);
        }
        double[] output = new double[dim1];
        for (int i = 0; i < dim1; i++) {
            output[i] = get(i,index);
        }
        return new DenseVector(output);
    }

    /**
     * Sets the column to the supplied vector value.
     * @param index The column to set.
     * @param vector The vector to write.
     */
    public void setColumn(int index, SGDVector vector) {
        if (index < 0 || index > dim2) {
            throw new IllegalArgumentException("Invalid column index, must be [0,"+dim2+"), received " + index);
        }
        if (vector.size() == dim1) {
            if (vector instanceof DenseVector) {
                for (int i = 0; i < dim1; i++) {
                    values[i][index] = vector.get(i);
                }
            } else {
                for (VectorTuple t : vector) {
                    values[t.index][index] = t.value;
                }
            }
        } else {
            throw new IllegalArgumentException("Vector size mismatch, expected " + dim1 + " found " + vector.size());
        }
    }

    /**
     * Calculates the sum of the specified row.
     * @param rowIndex The index of the row to sum.
     * @return The row sum.
     */
    public double rowSum(int rowIndex) {
        if (rowIndex < 0 || rowIndex > dim1) {
            throw new IllegalArgumentException("Invalid row index, must be [0,"+dim1+"), received " + rowIndex);
        }
        double sum = 0d;
        for (int i = 0; i < dim2; i++) {
            sum += get(rowIndex,i);
        }
        return sum;
    }

    /**
     * Calculates the sum of the specified column.
     * @param columnIndex The index of the column to sum.
     * @return The column sum.
     */
    public double columnSum(int columnIndex) {
        if (columnIndex < 0 || columnIndex > dim2) {
            throw new IllegalArgumentException("Invalid column index, must be [0,"+dim2+"), received " + columnIndex);
        }
        double sum = 0d;
        for (int i = 0; i < dim1; i++) {
            sum += get(i,columnIndex);
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

    /**
     * Returns a copy of this matrix as a 2d array.
     * @return A copy of this matrix.
     */
    public double[][] toArray() {
        double[][] copy = new double[dim1][];
        for (int i = 0; i < dim1; i++) {
            copy[i] = Arrays.copyOf(values[i],dim2);
        }
        return copy;
    }

    /**
     * Is this a square matrix?
     * @return True if the matrix is square.
     */
    public boolean isSquare() {
        return dim1 == dim2;
    }

    /**
     * Returns true if this matrix is square and symmetric.
     * @return True if the matrix is symmetric.
     */
    public boolean isSymmetric() {
        if (!isSquare()) {
            return false;
        } else {
            for (int i = 0; i < dim1; i++) {
                for (int j = i + 1; j < dim1; j++) {
                    if (Double.compare(get(i,j),get(j,i)) != 0) {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    /**
     * Computes the Cholesky factorization of a positive definite matrix.
     * <p>
     * If the matrix is not symmetric or positive definite then it returns an empty optional.
     * @return The Cholesky factorization or an empty optional.
     */
    public Optional<CholeskyFactorization> choleskyFactorization() {
        if (!isSymmetric()) {
            logger.fine("Returning empty optional as matrix is not symmetric");
            return Optional.empty();
        } else {
            // Copy the matrix first
            DenseMatrix chol = new DenseMatrix(this);
            double[][] cholMatrix = chol.values;

            // Compute factorization
            for (int i = 0; i < dim1; i++) {
                for (int j = i; j < dim1; j++) {
                    double sum = cholMatrix[i][j];
                    for (int k = i - 1; k >= 0; k--) {
                        sum -= cholMatrix[i][k] * cholMatrix[j][k];
                    }
                    if (i == j) {
                        if (sum <= FACTORIZATION_TOLERANCE) {
                            // Matrix is not positive definite as it has a negative diagonal element.
                            logger.fine("Returning empty optional as matrix is not positive definite");
                            return Optional.empty();
                        } else {
                            cholMatrix[i][i] = Math.sqrt(sum);
                        }
                    } else {
                        cholMatrix[j][i] = sum / cholMatrix[i][i];
                    }
                }
            }

            // Zero out the upper triangle
            for (int i = 0; i < dim1; i++) {
                for (int j = 0; j < i; j++) {
                    cholMatrix[j][i] = 0.0;
                }
            }

            return Optional.of(new CholeskyFactorization(chol));
        }
    }

    /**
     * Computes the LU factorization of a square matrix.
     * <p>
     * If the matrix is singular or not square it returns an empty optional.
     * @return The LU factorization or an empty optional.
     */
    public Optional<LUFactorization> luFactorization() {
        if (!isSquare()) {
            logger.fine("Returning empty optional as matrix is not square");
            return Optional.empty();
        } else {
            // Copy the matrix first & init variables
            DenseMatrix lu = new DenseMatrix(this);
            double[][] luMatrix = lu.values;
            int[] permutation = new int[dim1];
            boolean oddSwaps = false;
            for (int i = 0; i < dim1; i++) {
                permutation[i] = i;
            }

            // Decompose matrix
            for (int i = 0; i < dim1; i++) {
                double max = 0.0;
                int maxIdx = i;

                // Find max element
                for (int k = i; k < dim1; k++) {
                    double cur = Math.abs(luMatrix[k][i]);
                    if (cur > max) {
                        max = cur;
                        maxIdx = k;
                    }
                }

                if (max < FACTORIZATION_TOLERANCE) {
                    // zero diagonal element, matrix is singular
                    logger.fine("Returning empty optional as matrix is singular");
                    return Optional.empty();
                }

                // Pivot matrix if necessary
                if (maxIdx != i) {
                    // Update permutation array
                    int tmpIdx = permutation[maxIdx];
                    permutation[maxIdx] = permutation[i];
                    permutation[i] = tmpIdx;
                    oddSwaps = !oddSwaps;

                    // Swap rows
                    double[] tmpRow = luMatrix[maxIdx];
                    luMatrix[maxIdx] = luMatrix[i];
                    luMatrix[i] = tmpRow;
                }

                // Eliminate row
                for (int j = i + 1; j < dim1; j++) {
                    // Rescale lower triangle
                    luMatrix[j][i] /= luMatrix[i][i];

                    for (int k = i + 1; k < dim1; k++) {
                        luMatrix[j][k] -= luMatrix[j][i] * luMatrix[i][k];
                    }
                }
            }

            // Split into two matrices
            DenseMatrix l = new DenseMatrix(lu);
            DenseMatrix u = new DenseMatrix(lu);

            // Zero lower triangle of u
            for (int i = 0; i < dim1; i++) {
                Arrays.fill(u.values[i],0,i,0.0);
            }

            // Zero upper triangle of l and set diagonal to 1.
            for (int i = 0; i < dim1; i++) {
                for (int j = 0; j <= i; j++) {
                    if (i == j) {
                        l.values[i][j] = 1.0;
                    } else {
                        l.values[j][i] = 0.0;
                    }
                }
            }

            return Optional.of(new LUFactorization(l,u,permutation,oddSwaps));
        }
    }

    /**
     * Eigen decomposition of a symmetric matrix.
     * <p>
     * Non-symmetric matrices return an empty Optional as they may have complex eigenvalues, and
     * any matrix which exceeds the default number of QL iterations in the decomposition also
     * returns an empty Optional.
     * @return The eigen decomposition of a symmetric matrix, or an empty optional if it's not symmetric.
     */
    public Optional<EigenDecomposition> eigenDecomposition() {
        if (!isSymmetric()) {
            logger.fine("Returning empty optional as matrix is not symmetric");
            return Optional.empty();
        } else {
            // Copy the matrix first & init variables
            DenseMatrix transform = new DenseMatrix(this);
            double[][] transformValues = transform.values;

            // arrays for holding the tridiagonal form.
            double[] diagonal = new double[dim1];
            double[] offDiagonal = new double[dim1]; // first element is zero

            // First tridiagonalize the matrix via a Householder reduction

            // Copy last row into diagonal
            System.arraycopy(transformValues[dim1 - 1], 0, diagonal, 0, dim1);

            // Iterate up the matrix, reducing it
            for (int i = dim1-1; i > 0; i--) {
                // Accumulate scale along current diagonal
                double scale = 0.0;
                for (int k = 0; k < i; k++) {
                    scale += Math.abs(diagonal[k]);
                }

                double diagElement = 0.0;
                if (scale == 0.0) {
                    offDiagonal[i] = 0.0; // if scale is zero then diagonal[0...i-1] = 0
                    for (int j = 0; j < i; j++) {
                        // copy in new row
                        diagonal[j] = transformValues[i-1][j];
                        // zero row & column
                        transformValues[i][j] = 0.0;
                        transformValues[j][i] = 0.0;
                    }
                } else {
                    // Generate Householder vector
                    for (int k = 0; k < i; k++) {
                        final double tmp = diagonal[k] / scale;
                        diagElement += tmp * tmp;
                        diagonal[k] = tmp;
                        offDiagonal[k] = 0;
                    }
                    final double nextDiag = diagonal[i-1];
                    final double offDiag = nextDiag >= 0 ? -Math.sqrt(diagElement) : Math.sqrt(diagElement);

                    offDiagonal[i] = scale * offDiag;
                    diagElement -= offDiag * nextDiag;
                    diagonal[i-1] = nextDiag - offDiag;

                    // Transform the remaining vectors
                    for (int j = 0; j < i; j++) {
                        final double transDiag = diagonal[j];
                        // Write back to matrix
                        transformValues[j][i] = transDiag;
                        double transOffDiag = offDiagonal[j] + transformValues[j][j] * transDiag;

                        // Sum remaining column and update off diagonals
                        for (int k = j + 1; k < i; k++) {
                            double tmp = transformValues[k][j];
                            transOffDiag += tmp * diagonal[k];
                            offDiagonal[k] += tmp * transDiag;
                        }
                        offDiagonal[j] = transOffDiag;
                    }

                    double scaledElementSum = 0.0;
                    for (int j = 0; j < i; j++) {
                        final double tmp = offDiagonal[j] / diagElement;
                        offDiagonal[j] = tmp;
                        scaledElementSum += tmp * diagonal[j];
                    }
                    final double offDiagScalingFactor = scaledElementSum / (diagElement + diagElement);
                    for (int j = 0; j < i; j++) {
                        offDiagonal[j] -= offDiagScalingFactor * diagonal[j];
                    }

                    for (int j = 0; j < i; j++) {
                        final double tmpDiag = diagonal[j];
                        final double tmpOffDiag = offDiagonal[j];
                        for (int k = j; k < i; k++) {
                            transformValues[k][j] -= (tmpDiag * offDiagonal[k]) + (tmpOffDiag * diagonal[k]);
                        }
                        diagonal[j] = transformValues[i-1][j];
                        transformValues[i][j] = 0.0;
                    }
                }
                diagonal[i] = diagElement;
            }

            // Finish transformation to tridiagonal
            int dimMinusOne = dim1-1;
            for (int i = 0; i < dimMinusOne; i++) {
                transformValues[dimMinusOne][i] = transformValues[i][i];
                transformValues[i][i] = 1.0;
                final int nextIdx = i + 1;
                final double nextDiag = diagonal[nextIdx];
                if (nextDiag != 0.0) {
                    // Recompute diagonal and rescale matrix
                    for (int k = 0; k < nextIdx; k++) {
                        diagonal[k] = transformValues[k][nextIdx] / nextDiag;
                    }
                    for (int j = 0; j < nextIdx; j++) {
                        double scaleAccumulator = 0.0;
                        for (int k = 0; k < nextIdx; k++) {
                            scaleAccumulator += transformValues[k][nextIdx] * transformValues[k][j];
                        }
                        for (int k = 0; k < nextIdx; k++) {
                            transformValues[k][j] -= scaleAccumulator * diagonal[k];
                        }
                    }
                    // Zero lower column
                    for (int j = 0; j < nextIdx; j++) {
                        transformValues[j][nextIdx] = 0.0;
                    }
                }
            }
            for (int j = 0; j < dim1; j++) {
                diagonal[j] = transformValues[dimMinusOne][j];
                transformValues[dimMinusOne][j] = 0.0;
            }
            transformValues[dimMinusOne][dimMinusOne] = 1.0;
            offDiagonal[0] = 0.0;

            // Copy to dense vector/matrix for storage in the returned object as we're going to mutate these arrays
            DenseVector diagVector = DenseVector.createDenseVector(diagonal);
            DenseVector offDiagVector = DenseVector.createDenseVector(offDiagonal);
            DenseMatrix householderMatrix = new DenseMatrix(transform);

            // Then compute eigen vectors & values using an iterated tridiagonal QL algorithm

            // Setup constants
            final int maxItr = 35; // Maximum number of QL iterations before giving up and returning empty optional.
            final double eps = Double.longBitsToDouble(4372995238176751616L); // Math.pow(2,-52)

            // Copy off diagonal up for ease of use
            System.arraycopy(offDiagonal, 1, offDiagonal, 0, dimMinusOne);
            offDiagonal[dimMinusOne] = 0.0;

            double diagAccum = 0.0;
            double largestDiagSum = 0.0;
            for (int i = 0; i < dim1; i++) {
                largestDiagSum = Math.max(largestDiagSum, Math.abs(diagonal[i]) + Math.abs(offDiagonal[i]));
                final double testVal = largestDiagSum*eps;
                // Find small value to partition the matrix
                int idx = i;
                while (idx < dim1) {
                    if (Math.abs(offDiagonal[idx]) <= testVal) {
                        break;
                    }
                    idx++;
                }

                // if we didn't break out of the loop the diagonal value is an eigenvalue
                // otherwise perform QL iterations
                if (idx > i) {
                    int iter = 0;
                    do {
                        if (iter > maxItr) {
                            // Exceeded QL iteration count;
                            logger.fine("Exceeded QL iteration count in eigenDecomposition");
                            return Optional.empty();
                        } else {
                            iter++;
                        }

                        // Compute shift
                        final double curDiag = diagonal[i];
                        final double shift = (diagonal[i+1] - curDiag) / (2 * offDiagonal[i]);
                        final double shiftLength = shift < 0 ? -Math.hypot(shift, 1.0) : Math.hypot(shift, 1.0);
                        diagonal[i] = offDiagonal[i] / (shift + shiftLength);
                        diagonal[i+1] = offDiagonal[i] * (shift + shiftLength);

                        final double nextDiag = diagonal[i+1];
                        final double diagShift = curDiag - diagonal[i];
                        for (int j = i + 2; j < dim1; j++) {
                            diagonal[j] -= diagShift;
                        }
                        diagAccum += diagShift;

                        // Compute implicit QL
                        double partitionDiag = diagonal[idx];
                        final double oldOffDiag = offDiagonal[i+1];
                        double c = 1.0, c2 = 1.0, c3 = 1.0;
                        double s = 0.0, prevS = 0.0;
                        for (int j = idx-1; j >= i; j--) {
                            c3 = c2;
                            c2 = c;
                            prevS = s;
                            final double scaledOffDiag = c * offDiagonal[j];
                            final double scaledDiag = c * partitionDiag;
                            final double dist = Math.hypot(partitionDiag, offDiagonal[j]);
                            offDiagonal[j+1] = s * dist;
                            s = offDiagonal[j] / dist;
                            c = partitionDiag / dist;
                            partitionDiag = (c * diagonal[j]) - (s * scaledOffDiag);
                            diagonal[j+1] = scaledDiag + s * ((c * scaledOffDiag) + (s * diagonal[j]));

                            // Update eigenvectors
                            for (int k = 0; k < dim1; k++) {
                                final double[] row = transformValues[k];
                                final double tmp = row[j+1];
                                row[j+1] = (s * row[j]) + (c * tmp);
                                row[j] = (c * row[j]) - (s * tmp);
                            }
                        }
                        partitionDiag = -s * prevS * c3 * oldOffDiag * offDiagonal[i] / nextDiag;
                        offDiagonal[i] = s * partitionDiag;
                        diagonal[i] = c * partitionDiag;
                    } while (Math.abs(offDiagonal[i]) > testVal);
                }

                diagonal[i] += diagAccum;
                offDiagonal[i] = 0.0;
            }

            // Sort eigenvalues and eigenvectors
            int[] indices = SortUtil.argsort(diagonal, false);
            double[] eigenValues = new double[dim1];
            double[][] eigenVectors = new double[dim1][dim1];

            for (int i = 0; i < indices.length; i++) {
                eigenValues[i] = diagonal[indices[i]];
                for (int j = 0; j < dim1; j++) {
                    eigenVectors[j][i] = transformValues[j][indices[i]];
                }
            }

            return Optional.of(new EigenDecomposition(new DenseVector(eigenValues),new DenseMatrix(eigenVectors),diagVector,offDiagVector,householderMatrix));
        }
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
     * Returns a dense vector containing each column sum.
     * @return The column sums.
     */
    public DenseVector columnSum() {
        double[] columnSum = new double[dim2];
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                columnSum[j] += get(i,j);
            }
        }
        return new DenseVector(columnSum);
    }

    /**
     * Returns a new DenseMatrix containing a copy of the selected columns.
     * <p>
     * Throws {@link IllegalArgumentException} if any column index is invalid or the array is null/empty.
     * @param columnIndices The column indices
     * @return The submatrix comprising the selected columns.
     */
    public DenseMatrix selectColumns(int[] columnIndices) {
        if (columnIndices == null || columnIndices.length == 0) {
            throw new IllegalArgumentException("Invalid column indices.");
        }
        DenseMatrix returnVal = new DenseMatrix(dim1,columnIndices.length);

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < columnIndices.length; j++) {
                int curIdx = columnIndices[j];
                if (curIdx < 0 || curIdx >= dim2) {
                    throw new IllegalArgumentException("Invalid column index, expected [0, " + dim2 +"), found " + curIdx);
                }
                returnVal.values[i][j] = get(i,curIdx);
            }
        }

        return returnVal;
    }

    /**
     * Returns a new DenseMatrix containing a copy of the selected columns.
     * <p>
     * Throws {@link IllegalArgumentException} if any column index is invalid or the list is null/empty.
     * @param columnIndices The column indices
     * @return The submatrix comprising the selected columns.
     */
    public DenseMatrix selectColumns(List<Integer> columnIndices) {
        if (columnIndices == null || columnIndices.isEmpty()) {
            throw new IllegalArgumentException("Invalid column indices.");
        }
        DenseMatrix returnVal = new DenseMatrix(dim1,columnIndices.size());

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < columnIndices.size(); j++) {
                int curIdx = columnIndices.get(j);
                if (curIdx < 0 || curIdx >= dim2) {
                    throw new IllegalArgumentException("Invalid column index, expected [0, " + dim2 +"), found " + curIdx);
                }
                returnVal.values[i][j] = get(i,curIdx);
            }
        }

        return returnVal;
    }

    private class DenseMatrixIterator implements MatrixIterator {
        private final DenseMatrix matrix;
        private final MatrixTuple tuple;
        private int i;
        private int j;

        DenseMatrixIterator(DenseMatrix matrix) {
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

    /**
     * The output of a successful Cholesky factorization.
     * <p>
     * Essentially wraps a {@link DenseMatrix}, but has additional
     * operations which allow more efficient implementations when the
     * matrix is known to be the result of a Cholesky factorization.
     * <p>
     * Mutating the wrapped matrix will cause undefined behaviour in the methods
     * of this class.
     * <p>
     * May be refactored into a record in the future.
     */
    public static final class CholeskyFactorization implements Matrix.Factorization {
        private final DenseMatrix lMatrix;

        CholeskyFactorization(DenseMatrix lMatrix) {
            this.lMatrix = lMatrix;
        }

        /**
         * The lower triangular factorized matrix.
         * @return The factorization matrix.
         */
        public DenseMatrix lMatrix() {
            return lMatrix;
        }

        @Override
        public int dim1() {
            return lMatrix.dim1;
        }

        @Override
        public int dim2() {
            return lMatrix.dim2;
        }

        /**
         * Compute the matrix determinant of the factorized matrix.
         * @return The matrix determinant.
         */
        @Override
        public double determinant() {
            double det = 0.0;
            for (int i = 0; i < lMatrix.dim1; i++) {
                det *= lMatrix.values[i][i] * lMatrix.values[i][i];
            }
            return det;
        }

        /**
         * Solves a system of linear equations A * b = y, where y is the input vector,
         * A is the matrix which produced this Cholesky factorization, and b is the returned value.
         * @param vector The input vector y.
         * @return The vector b.
         */
        @Override
        public DenseVector solve(SGDVector vector) {
            if (vector.size() != lMatrix.dim1) {
                throw new IllegalArgumentException("Size mismatch, expected " + lMatrix.dim1 + ", received " + vector.size());
            }
            final double[] vectorArr = vector.toArray();
            final double[] output = new double[lMatrix.dim1];

            // Solve matrix . y = vector
            for (int i = 0; i < lMatrix.dim1; i++) {
                double sum = vectorArr[i];
                for (int j = i-1; j >= 0; j--) {
                    sum -= lMatrix.values[i][j] * output[j];
                }
                output[i] = sum / lMatrix.values[i][i];
            }

            // Solve matrix^T . output = y
            for (int i = lMatrix.dim1-1; i >= 0; i--) {
                double sum = output[i];
                for (int j = i+1; j < lMatrix.dim1; j++) {
                    sum -= lMatrix.values[j][i] * output[j];
                }
                output[i] = sum / lMatrix.values[i][i];
            }

            return new DenseVector(output);
        }

        /**
         * Solves the system A * X = B, where B is the input matrix, and A is the matrix which
         * produced this Cholesky factorization.
         * @param matrix The input matrix B.
         * @return The matrix X.
         */
        @Override
        public DenseMatrix solve(Matrix matrix) {
            if (matrix.getDimension1Size() != lMatrix.dim1) {
                throw new IllegalArgumentException("Size mismatch, expected " + lMatrix.dim1 + ", received " + matrix.getDimension1Size());
            }
            final int outputDim1 = lMatrix.dim1;
            final int outputDim2 = matrix.getDimension2Size();
            final DenseMatrix output = new DenseMatrix(matrix);
            final double[][] outputArr = output.values;

            // Solve L.Y = B
            for (int i = 0; i < outputDim1; i++) {
                for (int j = 0; j < outputDim2; j++) {
                    for (int k = 0; k < i; k++) {
                        outputArr[i][j] -= outputArr[k][j] * lMatrix.values[i][k];
                    }
                    // scale by diagonal
                    outputArr[i][j] /= lMatrix.values[i][i];
                }
            }

            // Solve L^T.X = Y
            for (int i = outputDim1 - 1; i >= 0; i--) {
                for (int j = 0; j < outputDim2; j++) {
                    for (int k = i + 1; k < outputDim2; k++) {
                        outputArr[i][j] -= outputArr[k][j] * lMatrix.values[k][i];
                    }
                    // scale by diagonal
                    outputArr[i][j] /= lMatrix.values[i][i];
                }
            }

            return output;
        }
    }

    /**
     * The output of a successful LU factorization.
     * <p>
     * Essentially wraps a pair of {@link DenseMatrix}, but has additional
     * operations which allow more efficient implementations when the
     * matrices are known to be the result of a LU factorization.
     * <p>
     * Mutating the wrapped matrices will cause undefined behaviour in the methods
     * of this class.
     * <p>
     * May be refactored into a record in the future.
     */
    public static final class LUFactorization implements Matrix.Factorization {
        private final DenseMatrix lower;
        private final DenseMatrix upper;
        private final int[] permutationArr;
        private final Matrix permutationMatrix;
        private final boolean oddSwaps;

        LUFactorization(DenseMatrix lower, DenseMatrix upper, int[] permutationArr, boolean oddSwaps) {
            this.lower = lower;
            this.upper = upper;
            this.permutationArr = permutationArr;
            SparseVector[] vecs = new SparseVector[permutationArr.length];
            for (int i = 0; i < vecs.length; i++) {
                vecs[i] = new SparseVector(lower.dim1,new int[]{permutationArr[i]}, new double[]{1.0});
            }
            this.permutationMatrix = DenseSparseMatrix.createFromSparseVectors(vecs);
            this.oddSwaps = oddSwaps;
        }

        /**
         * The lower triangular matrix, with ones on the diagonal.
         * @return The lower triangular matrix.
         */
        public DenseMatrix lower() {
            return lower;
        }

        /**
         * The upper triangular matrix.
         * @return The upper triangular matrix.
         */
        public DenseMatrix upper() {
            return upper;
        }

        /**
         * The row permutations applied to get this factorization.
         * @return The permutations.
         */
        public int[] permutationArr() {
            return permutationArr;
        }

        /**
         * The row permutations stored as a sparse matrix of ones.
         * @return A sparse matrix version of the permutations.
         */
        public Matrix permutationMatrix() {
            return permutationMatrix;
        }

        /**
         * Is there an odd number of row swaps (used to compute the determinant).
         * @return True if there is an odd number of swaps.
         */
        public boolean oddSwaps() {
            return oddSwaps;
        }

        @Override
        public int dim1() {
            return permutationArr.length;
        }

        @Override
        public int dim2() {
            return permutationArr.length;
        }

        /**
         * Compute the matrix determinant of the factorized matrix.
         * @return The matrix determinant.
         */
        @Override
        public double determinant() {
            double det = 0.0;
            for (int i = 0; i < upper.dim1; i++) {
                det *= upper.values[i][i];
            }
            if (oddSwaps) {
                return -det;
            } else {
                return det;
            }
        }

        /**
         * Solves a system of linear equations A * b = y, where y is the input vector,
         * A is the matrix which produced this LU factorization, and b is the returned value.
         * @param vector The input vector y.
         * @return The vector b.
         */
        @Override
        public DenseVector solve(SGDVector vector) {
            if (vector.size() != lower.dim1) {
                throw new IllegalArgumentException("Size mismatch, expected " + lower.dim1 + ", received " + vector.size());
            }
            // Apply permutation to input
            final double[] vectorArr = vector.toArray();
            final double[] output = new double[vectorArr.length];
            for (int i = 0; i < permutationArr.length; i++) {
                output[i] = vectorArr[permutationArr[i]];

                // Solve L * Y = b
                for (int k = 0; k < i; k++) {
                    output[i] -= lower.values[i][k] * output[k];
                }
            }

            // Solve U * X = Y
            for (int i = permutationArr.length-1; i >= 0; i--) {
                for (int k = i + 1; k < permutationArr.length; k++) {
                    output[i] -= upper.values[i][k] * output[k];
                }
                output[i] /= upper.values[i][i];
            }

            return new DenseVector(output);
        }

        /**
         * Solves the system A * X = Y, where Y is the input matrix, and A is the matrix which
         * produced this LU factorization.
         * @param matrix The input matrix Y.
         * @return The matrix X.
         */
        @Override
        public DenseMatrix solve(Matrix matrix) {
            if (matrix.getDimension1Size() != lower.dim1) {
                throw new IllegalArgumentException("Size mismatch, expected " + lower.dim1 + ", received " + matrix.getDimension1Size());
            }
            final int outputDim1 = lower.dim1;
            final int outputDim2 = matrix.getDimension2Size();
            final double[][] output = new double[lower.dim1][];

            // Apply permutation and copy over
            for (int i = 0; i < outputDim1; i++) {
                int permutedIdx = permutationArr[i];
                for (int j = 0; j < outputDim2; j++) {
                    output[i] = matrix.getRow(permutedIdx).toArray();
                }
            }

            // Solve LY = B
            for (int i = 0; i < outputDim1; i++) {
                for (int j = i + 1; j < outputDim1; j++) {
                    for (int k = 0; k < outputDim2; k++) {
                        output[j][k] -= output[i][k] * lower.values[j][i];
                    }
                }
            }

            // Solve UX = Y
            for (int i = outputDim1 - 1; i >= 0; i--) {
                // scale by diagonal
                for (int j = 0; j < outputDim2; j++) {
                    output[i][j] /= upper.values[i][i];
                }
                for (int j = 0; j < i; j++) {
                    for (int k = 0; k < outputDim2; k++) {
                        output[j][k] -= output[i][k] * upper.values[j][i];
                    }
                }
            }

            return new DenseMatrix(output);
        }
    }

    /**
     * The output of a successful eigen decomposition.
     * <p>
     * Wraps a dense vector containing the eigenvalues and a dense matrix containing the eigenvectors as columns.
     * Mutating these fields will cause undefined behaviour.
     * <p>
     * Also has fields representing the tridiagonal form used as an intermediate step in the eigen decomposition.
     * <p>
     * May be refactored into a record in the future.
     */
    public static final class EigenDecomposition implements Matrix.Factorization {
        // Eigen decomposition fields
        private final DenseVector eigenvalues;
        private final DenseMatrix eigenvectors;

        // Tridiagonal form fields
        private final DenseVector diagonal;
        private final DenseVector offDiagonal;
        private final DenseMatrix householderMatrix;

        EigenDecomposition(DenseVector eigenvalues, DenseMatrix eigenvectors, DenseVector diagonal, DenseVector offDiagonal, DenseMatrix householderMatrix) {
            this.eigenvalues = eigenvalues;
            this.eigenvectors = eigenvectors;
            this.diagonal = diagonal;
            this.offDiagonal = offDiagonal;
            this.householderMatrix = householderMatrix;
        }

        /**
         * The vector of eigenvalues, in descending order.
         * @return The eigenvalues.
         */
        public DenseVector eigenvalues() {
            return eigenvalues;
        }

        /**
         * The eigenvectors for each eigenvalue, stored in the columns of the matrix.
         * @return A matrix containing the eigenvalues as columns.
         */
        public DenseMatrix eigenvectors() {
            return eigenvectors;
        }

        /**
         * The diagonal vector of the tridiagonal form.
         * @return The diagonal vector.
         */
        public DenseVector diagonal() {
            return diagonal;
        }

        /**
         * The off diagonal vector, with the first element set to zero.
         * @return The off diagonal vector.
         */
        public DenseVector offDiagonal() {
            return offDiagonal;
        }

        /**
         * The Householder matrix produced during the tridiagonalisation.
         * @return The Householder matrix.
         */
        public DenseMatrix householderMatrix() {
            return householderMatrix;
        }

        @Override
        public int dim1() {
            return eigenvalues.size();
        }

        @Override
        public int dim2() {
            return eigenvalues.size();
        }

        /**
         * Computes the determinant of the matrix which was decomposed.
         * <p>
         * This is the product of the eigenvalues.
         * @return The determinant.
         */
        @Override
        public double determinant() {
            return eigenvalues.reduce(1.0,DoubleUnaryOperator.identity(), (a,b) -> a*b);
        }

        /**
         * Returns true if all the eigenvalues are positive.
         * @return True if the eigenvalues are positive.
         */
        public boolean positiveEigenvalues() {
            return eigenvalues.reduce(true,DoubleUnaryOperator.identity(),(value, bool) -> bool && value > 0.0);
        }

        /**
         * Returns true if all the eigenvalues are non-zero.
         * @return True if the eigenvalues are non-zero (i.e. the matrix is not singular).
         */
        public boolean nonSingular() {
            return eigenvalues.reduce(true,DoubleUnaryOperator.identity(),(value, bool) -> bool && value != 0.0);
        }

        /**
         * Returns the dense vector representing the i'th eigenvector.
         * @param i The index.
         * @return The i'th eigenvector.
         */
        public DenseVector getEigenVector(int i) {
            if (i < 0 || i > eigenvectors.dim1) {
                throw new IllegalArgumentException("Invalid index, must be [0," + eigenvectors.dim1 + "), found " + i);
            }
            return eigenvectors.getColumn(i);
        }

        /**
         * Solves a system of linear equations A * b = y, where y is the input vector,
         * A is the matrix which produced this eigen decomposition, and b is the returned value.
         * @param vector The input vector y.
         * @return The vector b.
         */
        @Override
        public DenseVector solve(SGDVector vector) {
            if (vector.size() != eigenvectors.dim1) {
                throw new IllegalArgumentException("Size mismatch, expected " + eigenvectors.dim1 + ", received " + vector.size());
            }
            final double[] output = new double[vector.size()];
            for (int i = 0; i < output.length; i++) {
                DenseVector eigenVector = getEigenVector(i);
                double value = vector.dot(eigenVector) / eigenvalues.get(i);
                for (int j = 0; j < output.length; j++) {
                    output[j] += value * eigenVector.get(j);
                }
            }

            return new DenseVector(output);
        }

        /**
         * Solves the system A * X = Y, where Y is the input matrix, and A is the matrix which
         * produced this eigen decomposition.
         * @param matrix The input matrix Y.
         * @return The matrix X.
         */
        @Override
        public DenseMatrix solve(Matrix matrix) {
            if (matrix.getDimension1Size() != eigenvectors.dim1) {
                throw new IllegalArgumentException("Size mismatch, expected " + eigenvectors.dim1 + ", received " + matrix.getDimension1Size());
            }
            final int outputDim1 = eigenvalues.size();
            final int outputDim2 = matrix.getDimension2Size();
            final double[][] output = new double[outputDim1][outputDim2];

            for (int k = 0; k < outputDim2; k++) {
                SGDVector column = matrix.getColumn(k);
                for (int i = 0; i < outputDim1; i++) {
                    DenseVector eigen = getEigenVector(i);
                    double value = eigen.dot(column) / eigenvalues.get(i);
                    for (int j = 0; j < output.length; j++) {
                        output[j][k] += value * eigen.get(j);
                    }
                }
            }

            return new DenseMatrix(output);
        }
    }
}
