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

package org.tribuo.math.optimisers.util;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.MatrixIterator;
import org.tribuo.math.la.MatrixTuple;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.math.protos.DenseTensorProto;
import org.tribuo.math.protos.ShrinkingDenseTensorProto;
import org.tribuo.math.protos.TensorProto;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.util.function.DoubleUnaryOperator;

/**
 * A subclass of {@link DenseMatrix} which shrinks the value every time a new value is added.
 * <p>
 * Be careful when modifying this or {@link DenseMatrix}.
 */
public class ShrinkingMatrix extends DenseMatrix implements ShrinkingTensor {
    private final double baseRate;
    private final double lambdaSqrt;
    private final boolean scaleShrinking;
    private final boolean reproject;
    private double squaredTwoNorm;
    private int iteration;
    private double multiplier;

    /**
     * Constructs a shrinking matrix copy of the supplied dense matrix.
     * <p>
     * This matrix shrinks during each call to {@link #intersectAndAddInPlace(Tensor, DoubleUnaryOperator)}.
     * @param v The matrix to copy.
     * @param baseRate The base amount of shrinking to apply after each update.
     * @param scaleShrinking If true reduce the shrinking value over time proportionally to the number of updates.
     */
    public ShrinkingMatrix(DenseMatrix v, double baseRate, boolean scaleShrinking) {
        super(v);
        this.baseRate = baseRate;
        this.scaleShrinking = scaleShrinking;
        this.lambdaSqrt = 0.0;
        this.reproject = false;
        this.squaredTwoNorm = 0.0;
        this.iteration = 1;
        this.multiplier = 1.0;
    }

    /**
     * Constructs a shrinking matrix copy of the supplied dense matrix.
     * <p>
     * This matrix shrinks during each call to {@link #intersectAndAddInPlace(Tensor, DoubleUnaryOperator)},
     * and then reprojects the matrix so it has the same twoNorm.
     * @param v The matrix to copy.
     * @param baseRate The base rate of shrinkage.
     * @param lambda The lambda value (see {@link org.tribuo.math.optimisers.Pegasos}).
     */
    public ShrinkingMatrix(DenseMatrix v, double baseRate, double lambda) {
        super(v);
        this.baseRate = baseRate;
        this.scaleShrinking = true;
        this.lambdaSqrt = Math.sqrt(lambda);
        this.reproject = true;
        this.squaredTwoNorm = 0.0;
        this.iteration = 1;
        this.multiplier = 1.0;
    }

    /**
     * Deserialization constructor.
     * @param v The dense matrix.
     * @param baseRate Base rate of shrinkage.
     * @param scaleShrinking Is the scale shrinking?
     * @param lambdaSqrt sqrt of the lambda parameter.
     * @param reproject Should the matrix reproject itself?
     * @param squaredTwoNorm The squaredTwoNorm of the matrix.
     * @param iteration The iteration number.
     * @param multiplier The current multiplier
     */
    private ShrinkingMatrix(DenseMatrix v, double baseRate, boolean scaleShrinking, double lambdaSqrt, boolean reproject, double squaredTwoNorm, int iteration, double multiplier) {
        super(v);
        this.baseRate = baseRate;
        this.scaleShrinking = scaleShrinking;
        this.lambdaSqrt = lambdaSqrt;
        if (!reproject && lambdaSqrt != 0.0) {
            throw new IllegalStateException("Invalid ShrinkingMatrix, when reproject is true lambda must be zero");
        }
        this.reproject = reproject;
        this.squaredTwoNorm = squaredTwoNorm;
        if (iteration < 0) {
            throw new IllegalArgumentException("Invalid ShrinkingMatrix, iteration must be non-negative");
        }
        this.iteration = iteration;
        this.multiplier = multiplier;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static ShrinkingMatrix deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        ShrinkingDenseTensorProto proto = message.unpack(ShrinkingDenseTensorProto.class);
        DenseMatrix data = DenseMatrix.unpackProto(proto.getData());
        return new ShrinkingMatrix(data, proto.getBaseRate(), proto.getScaleShrinking(), proto.getLambdaSqrt(),
                proto.getReproject(), proto.getSquaredTwoNorm(), proto.getIteration(), proto.getMultiplier());
    }

    @Override
    public TensorProto serialize() {
        TensorProto.Builder builder = TensorProto.newBuilder();

        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(ShrinkingMatrix.class.getName());

        ShrinkingDenseTensorProto.Builder shrinkingBuilder = ShrinkingDenseTensorProto.newBuilder();
        DenseTensorProto.Builder dataBuilder = DenseTensorProto.newBuilder();
        dataBuilder.addDimensions(dim1);
        dataBuilder.addDimensions(dim2);
        ByteBuffer buffer = ByteBuffer.allocate(dim1 * dim2 * 8).order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer doubleBuffer = buffer.asDoubleBuffer();
        for (int i = 0; i < values.length; i ++) {
            doubleBuffer.put(values[i]);
        }
        doubleBuffer.rewind();
        dataBuilder.setValues(ByteString.copyFrom(buffer));
        shrinkingBuilder.setData(dataBuilder.build());
        shrinkingBuilder.setBaseRate(baseRate);
        shrinkingBuilder.setLambdaSqrt(lambdaSqrt);
        shrinkingBuilder.setScaleShrinking(scaleShrinking);
        shrinkingBuilder.setReproject(reproject);
        shrinkingBuilder.setSquaredTwoNorm(squaredTwoNorm);
        shrinkingBuilder.setIteration(iteration);
        shrinkingBuilder.setMultiplier(multiplier);
        builder.setSerializedData(Any.pack(shrinkingBuilder.build()));

        return builder.build();
    }

    @Override
    public DenseMatrix convertToDense() {
        return new DenseMatrix(this);
    }

    @Override
    public DenseVector leftMultiply(SGDVector input) {
        if (input.size() == dim2) {
            double[] output = new double[dim1];
            for (VectorTuple tuple : input) {
                for (int i = 0; i < output.length; i++) {
                    output[i] += get(i, tuple.index) * tuple.value;
                }
            }

            return DenseVector.createDenseVector(output);
        } else {
            throw new IllegalArgumentException("input.size() != dim2");
        }
    }

    @Override
    public void intersectAndAddInPlace(Tensor other, DoubleUnaryOperator f) {
        if (other instanceof Matrix) {
            Matrix otherMat = (Matrix) other;
            if ((dim1 == otherMat.getDimension1Size()) && (dim2 == otherMat.getDimension2Size())) {
                double shrinkage = scaleShrinking ? 1.0 - (baseRate / iteration) : 1.0 - baseRate;
                scaleInPlace(shrinkage);
                for (MatrixTuple tuple : otherMat) {
                    double update = f.applyAsDouble(tuple.value);
                    double oldValue = values[tuple.i][tuple.j] * multiplier;
                    double newValue = oldValue + update;
                    squaredTwoNorm -= oldValue * oldValue;
                    squaredTwoNorm += newValue * newValue;
                    values[tuple.i][tuple.j] = newValue / multiplier;
                }
                if (reproject) {
                    double projectionNormaliser = (1.0 / lambdaSqrt) / twoNorm();
                    if (projectionNormaliser < 1.0) {
                        scaleInPlace(projectionNormaliser);
                    }
                }
                iteration++;
            } else {
                throw new IllegalStateException("Matrices are not the same size, this(" + dim1 + "," + dim2 + "), other(" + otherMat.getDimension1Size() + "," + otherMat.getDimension2Size() + ")");
            }
        } else {
            throw new IllegalStateException("Adding a non-Matrix to a Matrix");
        }
    }

    @Override
    public double get(int i, int j) {
        return values[i][j] * multiplier;
    }

    @Override
    public void scaleInPlace(double value) {
        multiplier *= value;
        if (Math.abs(multiplier) < tolerance) {
            reifyMultiplier();
        }
    }

    private void reifyMultiplier() {
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                values[i][j] *= multiplier;
            }
        }
        multiplier = 1.0;
    }

    @Override
    public double twoNorm() {
        return Math.sqrt(squaredTwoNorm);
    }

    @Override
    public MatrixIterator iterator() {
        return new ShrinkingMatrixIterator(this);
    }

    private class ShrinkingMatrixIterator implements MatrixIterator {
        private final ShrinkingMatrix matrix;
        private final MatrixTuple tuple;
        private int i;
        private int j;

        public ShrinkingMatrixIterator(ShrinkingMatrix matrix) {
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
            tuple.i = i;
            tuple.j = j;
            tuple.value = matrix.get(i, j);
            if (j < dim2 - 1) {
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

