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
import org.tribuo.math.protos.SparseTensorProto;
import org.tribuo.math.protos.TensorProto;
import org.tribuo.util.Util;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;

/**
 * A matrix which is dense in the first dimension and sparse in the second.
 * <p>
 * Backed by an array of {@link SparseVector}.
 */
public class DenseSparseMatrix implements Matrix {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final SparseVector[] values;
    private final int dim1;
    private final int dim2;
    private final int[] shape;

    /**
     * Constructs a DenseSparseMatrix from the supplied vector array.
     * <p>
     * Does not copy the values, used internally by the la package.
     * @param values The sparse vectors.
     */
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
        Arrays.fill(values, emptyVector);
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

    /**
     * Creates an identity matrix of the specified size.
     * @param dimension The matrix dimension.
     * @return The identity matrix.
     */
    public static DenseSparseMatrix createIdentity(int dimension) {
        SparseVector[] newValues = new SparseVector[dimension];
        for (int i = 0; i < dimension; i++) {
            newValues[i] = new SparseVector(dimension, new int[]{i}, new double[]{1.0});
        }
        return new DenseSparseMatrix(newValues);
    }

    /**
     * Creates a diagonal matrix using the supplied values.
     * @param diagonal The values along the diagonal.
     * @return A diagonal matrix.
     */
    public static DenseSparseMatrix createDiagonal(SGDVector diagonal) {
        int dimension = diagonal.size();
        SparseVector[] newValues = new SparseVector[dimension];
        for (int i = 0; i < dimension; i++) {
            newValues[i] = new SparseVector(dimension, new int[]{i}, new double[]{diagonal.get(i)});
        }
        return new DenseSparseMatrix(newValues);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static DenseSparseMatrix deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        SparseTensorProto proto = message.unpack(SparseTensorProto.class);
        List<Integer> shapeList = proto.getDimensionsList();
        int[] shape = Util.toPrimitiveInt(shapeList);
        if (shape.length != 2) {
            throw new IllegalArgumentException("Invalid proto, expected a vector, found shape " + Arrays.toString(shape));
        }
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 1) {
                throw new IllegalArgumentException("Invalid proto, shape must be positive, found " + shape[i] + " at position " + i);
            }
        }
        int numElements = proto.getNumNonZero();
        IntBuffer indicesBuffer = proto.getIndices().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
        if (indicesBuffer.remaining() != numElements * 2) {
            throw new IllegalArgumentException("Invalid proto, claimed " + (numElements * 2) + ", but only had " + indicesBuffer.remaining() + " indices");
        }
        DoubleBuffer valuesBuffer = proto.getValues().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        if (valuesBuffer.remaining() != numElements) {
            throw new IllegalArgumentException("Invalid proto, claimed " + numElements + ", but only had " + valuesBuffer.remaining() + " values");
        }
        SparseVector[] vectors = new SparseVector[shape[0]];
        List<Integer> indices = new ArrayList<>();
        List<Double> values = new ArrayList<>();
        int rowCounter = 0;
        for (int i = 0; i < numElements; i++) {
            int curI = indicesBuffer.get();
            int curJ = indicesBuffer.get();
            double curValue = valuesBuffer.get();
            while (curI != rowCounter) {
                vectors[rowCounter] = SparseVector.createAndValidate(shape[1],Util.toPrimitiveInt(indices),Util.toPrimitiveDouble(values));
                indices.clear();
                values.clear();
                rowCounter++;
            }
            indices.add(curJ);
            values.add(curValue);
        }
        vectors[rowCounter] = SparseVector.createAndValidate(shape[1],Util.toPrimitiveInt(indices),Util.toPrimitiveDouble(values));
        indices.clear();
        values.clear();
        rowCounter++;
        while (rowCounter < shape[0]) {
            vectors[rowCounter] = new SparseVector(shape[1],new int[0],new double[0]);
            rowCounter++;
        }
        return new DenseSparseMatrix(vectors);
    }

    @Override
    public TensorProto serialize() {
        SparseTensorProto.Builder dataBuilder = SparseTensorProto.newBuilder();
        dataBuilder.addAllDimensions(Arrays.stream(shape).boxed().collect(Collectors.toList()));
        int numNonZero = 0;
        for (int i = 0; i < values.length; i++) {
            numNonZero += values[i].numActiveElements();
        }
        ByteBuffer indicesBuffer = ByteBuffer.allocate(numNonZero * 2 * 4).order(ByteOrder.LITTLE_ENDIAN);
        IntBuffer intBuffer = indicesBuffer.asIntBuffer();
        ByteBuffer valuesBuffer = ByteBuffer.allocate(numNonZero * 8).order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer doubleBuffer = valuesBuffer.asDoubleBuffer();
        for (MatrixTuple i : this) {
            intBuffer.put(i.i);
            intBuffer.put(i.j);
            doubleBuffer.put(i.value);
        }
        intBuffer.rewind();
        doubleBuffer.rewind();
        dataBuilder.setIndices(ByteString.copyFrom(indicesBuffer));
        dataBuilder.setValues(ByteString.copyFrom(valuesBuffer));
        dataBuilder.setNumNonZero(numNonZero);

        TensorProto.Builder builder = TensorProto.newBuilder();

        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(DenseSparseMatrix.class.getName());
        builder.setSerializedData(Any.pack(dataBuilder.build()));

        return builder.build();
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
        if (i < 0 || i > dim1) {
            throw new IllegalArgumentException("Invalid row index, must be [0,"+dim1+"), received " + i);
        }
        return values[i];
    }

    /**
     * Gets a copy of the column.
     * <p>
     * This function is O(dim1 * log(dim2)) as it requires searching each vector for the column index.
     * @param i The column index.
     * @return A copy of the column as a sparse vector.
     */
    @Override
    public SparseVector getColumn(int i) {
        if (i < 0 || i > dim2) {
            throw new IllegalArgumentException("Invalid column index, must be [0,"+dim2+"), received " + i);
        }
        List<Integer> indexList = new ArrayList<>();
        List<Double> valueList = new ArrayList<>();
        for (int j = 0; j < dim1; j++) {
            double tmp = values[j].get(i);
            if (tmp != 0) {
                indexList.add(j);
                valueList.add(tmp);
            }
        }

        int[] indicesArr = new int[valueList.size()];
        double[] valuesArr = new double[valueList.size()];
        for (int j = 0; j < valueList.size(); j++) {
            indicesArr[j] = indexList.get(j);
            valuesArr[j] = valueList.get(j);
        }
        return new SparseVector(dim1, indicesArr, valuesArr);
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

        DenseSparseMatrixIterator(DenseSparseMatrix matrix) {
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
