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

package org.tribuo.math.optimisers;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.math.Parameters;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.MatrixIterator;
import org.tribuo.math.la.MatrixTuple;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.la.VectorIterator;
import org.tribuo.math.la.VectorTuple;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;
import java.util.logging.Logger;

/**
 * An implementation of the AdaGrad gradient optimiser with regularized dual averaging.
 * <p>
 * This gradient optimiser rewrites all the {@link Tensor}s in the {@link Parameters}
 * with {@link AdaGradRDATensor}. This means it keeps a different value in the {@link Tensor}
 * to the one produced when you call get(), so it can correctly apply regularisation to the parameters.
 * When {@link AdaGradRDA#finalise()} is called it rewrites the {@link Parameters} with standard dense {@link Tensor}s.
 * Follows the implementation in Factorie.
 * <p>
 * See:
 * <pre>
 * Duchi, J., Hazan, E., and Singer, Y.
 * "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
 * Journal of Machine Learning Research, 2012, 2121-2159.
 * </pre>
 */
public class AdaGradRDA implements StochasticGradientOptimiser {
    private static final Logger logger = Logger.getLogger(AdaGradRDA.class.getName());

    @Config(mandatory = true,description="Initial learning rate used to scale the gradients.")
    private double initialLearningRate;

    @Config(description="Epsilon for numerical stability around zero.")
    private double epsilon = 1e-6;

    @Config(description="l1 regularization penalty.")
    private double l1 = 0;

    @Config(description="l2 regularization penalty.")
    private double l2 = 0;

    @Config(description="Number of examples to scale the l1 and l2 penalties by.")
    private int numExamples = 1;

    private Parameters parameters = null;

    /**
     * Creates an AdaGradRDA optimiser with the specified parameter values.
     * @param initialLearningRate The learning rate.
     * @param epsilon The epsilon value for stabilising the gradient inversion.
     * @param l1 The l1 penalty.
     * @param l2 The l2 penalty.
     * @param numExamples The number of examples to scale the l1 and l2 penalties.
     */
    public AdaGradRDA(double initialLearningRate, double epsilon, double l1, double l2, int numExamples) {
        this.initialLearningRate = initialLearningRate;
        this.epsilon = epsilon;
        this.l1 = l1;
        this.l2 = l2;
        this.numExamples = numExamples;
    }

    /**
     * Creates an AdaGradRDA optimiser with the specified parameter values.
     * <p>
     * Sets the regularisation parameters to zero.
     * @param initialLearningRate The learning rate.
     * @param epsilon The epsilon value for stabilising the gradient inversion.
     */
    public AdaGradRDA(double initialLearningRate, double epsilon) {
        this(initialLearningRate,epsilon,0,0,1);
    }

    /**
     * For OLCUT.
     */
    private AdaGradRDA() { }

    @Override
    public void initialise(Parameters parameters) {
        this.parameters = parameters;
        Tensor[] curParams = parameters.get();
        Tensor[] newParams = new Tensor[curParams.length];
        for (int i = 0; i < newParams.length; i++) {
            if (curParams[i] instanceof DenseVector) {
                newParams[i] = new AdaGradRDAVector(((DenseVector) curParams[i]), initialLearningRate, epsilon, l1 / numExamples, l2 / numExamples);
            } else if (curParams[i] instanceof DenseMatrix) {
                newParams[i] = new AdaGradRDAMatrix(((DenseMatrix) curParams[i]), initialLearningRate, epsilon, l1 / numExamples, l2 / numExamples);
            } else {
                throw new IllegalStateException("Unknown Tensor subclass");
            }
        }
        parameters.set(newParams);
    }

    @Override
    public Tensor[] step(Tensor[] updates, double weight) {
        for (Tensor update : updates) {
            update.scaleInPlace(weight);
        }

        return updates;
    }

    @Override
    public void finalise() {
        Tensor[] curParams = parameters.get();
        Tensor[] newParams = new Tensor[curParams.length];
        for (int i = 0; i < newParams.length; i++) {
            if (curParams[i] instanceof AdaGradRDATensor) {
                newParams[i] = ((AdaGradRDATensor) curParams[i]).convertToDense();
            } else {
                throw new IllegalStateException("Finalising a Parameters which wasn't initialised with AdaGradRDA");
            }
        }
        parameters.set(newParams);
    }

    @Override
    public String toString() {
        return "AdaGradRDA(initialLearningRate="+initialLearningRate+",epsilon="+epsilon+",l1="+l1+",l2="+l2+")";
    }

    @Override
    public void reset() {
        parameters = null;
    }

    @Override
    public AdaGradRDA copy() {
        return new AdaGradRDA(initialLearningRate,epsilon,l1,l2,numExamples);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"StochasticGradientOptimiser");
    }

    /**
     * An interface which tags a {@link Tensor} with a convertToDense method.
     */
    private static interface AdaGradRDATensor {
        /**
         * Returns a dense copy of this tensor.
         * @return A dense copy.
         */
        public Tensor convertToDense();

        /**
         * Zeros the value around the threshold.
         * @param input The value to truncate.
         * @param threshold The threshold to zero.
         * @return The input value with the region around the threshold removed.
         */
        public static double truncate(double input, double threshold) {
            if (input > threshold) {
                return input - threshold;
            } else if (input < -threshold) {
                return input + threshold;
            } else {
                return 0.0;
            }
        }

    }

    /**
     * A subclass of {@link DenseVector} which uses {@link AdaGradRDATensor#truncate(double, double)} to
     * produce the values.
     * <p>
     * Be careful when modifying this or {@link DenseVector}.
     */
    private static class AdaGradRDAVector extends DenseVector implements AdaGradRDATensor {
        private final double learningRate;
        private final double epsilon;
        private final double l1;
        private final double l2;
        private final double[] gradSquares;
        private int iteration;

        public AdaGradRDAVector(DenseVector v, double learningRate, double epsilon, double l1, double l2) {
            super(v);
            this.learningRate = learningRate;
            this.epsilon = epsilon;
            this.l1 = l1;
            this.l2 = l2;
            this.gradSquares = new double[v.size()];
            this.iteration = 0;
        }

        private AdaGradRDAVector(double[] values, double learningRate, double epsilon, double l1, double l2, double[] gradSquares, int iteration) {
            super(values);
            this.learningRate = learningRate;
            this.epsilon = epsilon;
            this.l1 = l1;
            this.l2 = l2;
            this.gradSquares = gradSquares;
            this.iteration = iteration;
        }

        @Override
        public DenseVector convertToDense() {
            return DenseVector.createDenseVector(toArray());
        }

        @Override
        public AdaGradRDAVector copy() {
            return new AdaGradRDAVector(Arrays.copyOf(elements,elements.length),learningRate,epsilon,l1,l2,Arrays.copyOf(gradSquares,gradSquares.length),iteration);
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
            if (gradSquares[index] == 0.0) {
                return elements[index];
            } else {
                double h = ((Math.sqrt(gradSquares[index]) + epsilon) / learningRate) + iteration * l2;
                //double h = (1.0/learningRate) * (Math.sqrt(gradSquares[index]) + epsilon) + iteration*l2;
                double rate = 1.0/h;
                return rate * AdaGradRDATensor.truncate(elements[index], iteration*l1);
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

        @Override
        public void intersectAndAddInPlace(Tensor other, DoubleUnaryOperator f) {
            iteration++;
            SGDVector otherVec = (SGDVector) other;
            for (VectorTuple tuple : otherVec) {
                double update = f.applyAsDouble(tuple.value);
                elements[tuple.index] += update;
                gradSquares[tuple.index] += update*update;
            }
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
        public double dot(SGDVector other) {
            double score = 0.0;

            for (VectorTuple tuple : other) {
                score += get(tuple.index) * tuple.value;
            }

            return score;
        }

        @Override
        public VectorIterator iterator() {
            return new RDAVectorIterator(this);
        }

        private static class RDAVectorIterator implements VectorIterator {
            private final AdaGradRDAVector vector;
            private final VectorTuple tuple;
            private int index;

            public RDAVectorIterator(AdaGradRDAVector vector) {
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

    /**
     * A subclass of {@link DenseMatrix} which uses {@link AdaGradRDATensor#truncate(double, double)} to
     * produce the values.
     * <p>
     * Be careful when modifying this or {@link DenseMatrix}.
     */
    private static class AdaGradRDAMatrix extends DenseMatrix implements AdaGradRDATensor {
        private final double learningRate;
        private final double epsilon;
        private final double l1;
        private final double l2;
        private final double[][] gradSquares;
        private int iteration;

        public AdaGradRDAMatrix(DenseMatrix v, double learningRate, double epsilon, double l1, double l2) {
            super(v);
            this.learningRate = learningRate;
            this.epsilon = epsilon;
            this.l1 = l1;
            this.l2 = l2;
            this.gradSquares = new double[v.getDimension1Size()][v.getDimension2Size()];
            this.iteration = 0;
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
                        output[i] += get(i,tuple.index) * tuple.value;
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
                    for (MatrixTuple tuple : otherMat) {
                        double update = f.applyAsDouble(tuple.value);
                        values[tuple.i][tuple.j] += update;
                        gradSquares[tuple.i][tuple.j] += update*update;
                    }
                } else {
                    throw new IllegalStateException("Matrices are not the same size, this("+dim1+","+dim2+"), other("+otherMat.getDimension1Size()+","+otherMat.getDimension2Size()+")");
                }
            } else {
                throw new IllegalStateException("Adding a non-Matrix to a Matrix");
            }
        }

        @Override
        public double get(int i, int j) {
            if (gradSquares[i][j] == 0.0) {
                return values[i][j];
            } else {
                double h = ((Math.sqrt(gradSquares[i][j]) + epsilon) / learningRate) + iteration * l2;
                //double h = (1.0/learningRate) * (Math.sqrt(gradSquares[index]) + epsilon) + iteration*l2;
                double rate = 1.0/h;
                return rate * AdaGradRDATensor.truncate(values[i][j], iteration*l1);
            }
        }

        @Override
        public MatrixIterator iterator() {
            return new RDAMatrixIterator(this);
        }

        private static class RDAMatrixIterator implements MatrixIterator {
            private final AdaGradRDAMatrix matrix;
            private final MatrixTuple tuple;
            private final int dim2;
            private int i;
            private int j;

            public RDAMatrixIterator(AdaGradRDAMatrix matrix) {
                this.matrix = matrix;
                this.tuple = new MatrixTuple();
                this.dim2 = matrix.dim2;
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
                tuple.value = matrix.get(i,j);
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
}

