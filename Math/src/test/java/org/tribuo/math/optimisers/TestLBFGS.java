/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import org.junit.jupiter.api.Test;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestLBFGS {

    /**
     * Minimises a quadratic bowl: f(x) = 0.5 * x^T A x - b^T x.
     * <p>
     * For SPD A, the unique minimiser is x* = A^{-1} b.
     */
    @Test
    public void testQuadraticObjectiveConverges() {
        // 2D SPD matrix A and vector b.
        // A = [[4,1],[1,3]]; b = [1,2]
        // x* = A^{-1} b = [1/11, 7/11]
        final double a11 = 4.0, a12 = 1.0, a22 = 3.0;
        final double b1 = 1.0, b2 = 2.0;
        final double xOpt1 = 1.0 / 11.0;
        final double xOpt2 = 7.0 / 11.0;

        Parameters parameters = new VectorParameters(DenseVector.createDenseVector(new double[]{10.0, -10.0}));

        LBFGS optimiser = new LBFGS(10, 200, 1e-12, 1e-8);

        optimiser.optimize(parameters,
                (Tensor[] p) -> {
                    DenseVector x = (DenseVector) p[0];
                    double x1 = x.get(0);
                    double x2 = x.get(1);

                    // loss = 0.5 * x^T A x - b^T x
                    double quad = (a11 * x1 * x1) + (2.0 * a12 * x1 * x2) + (a22 * x2 * x2);
                    double loss = 0.5 * quad - (b1 * x1 + b2 * x2);

                    // gradient = A x - b
                    double g1 = a11 * x1 + a12 * x2 - b1;
                    double g2 = a12 * x1 + a22 * x2 - b2;
                    return new LBFGS.GradAndLoss(new Tensor[]{DenseVector.createDenseVector(new double[]{g1, g2})}, loss);
                },
                (Tensor[] p) -> {
                    DenseVector x = (DenseVector) p[0];
                    double x1 = x.get(0);
                    double x2 = x.get(1);
                    double quad = (a11 * x1 * x1) + (2.0 * a12 * x1 * x2) + (a22 * x2 * x2);
                    return 0.5 * quad - (b1 * x1 + b2 * x2);
                });

        DenseVector finalX = (DenseVector) parameters.get()[0];

        // Sanity check: we should be close to the known optimum.
        assertEquals(xOpt1, finalX.get(0), 1e-3);
        assertEquals(xOpt2, finalX.get(1), 1e-3);

        // Also ensure we're at/near the minimum by checking gradient norm.
        double g1 = a11 * finalX.get(0) + a12 * finalX.get(1) - b1;
        double g2 = a12 * finalX.get(0) + a22 * finalX.get(1) - b2;
        assertTrue(Math.sqrt(g1 * g1 + g2 * g2) < 1e-2);
    }

    /**
     * Minimises the Rosenbrock function:
     * f(x,y) = (a-x)^2 + b (y-x^2)^2.
     * <p>
     * Global minimum at (x,y) = (a, a^2) which is (1,1) for the standard choice.
     */
    @Test
    public void testRosenbrockConverges() {
        final double a = 1.0;
        final double b = 100.0;

        // Standard challenging start point.
        Parameters parameters = new VectorParameters(DenseVector.createDenseVector(new double[]{-1.2, 1.0}));

        // Tight tolerances and enough iterations to converge inside the narrow valley.
        LBFGS optimiser = new LBFGS(10, 2000, 1e-14, 1e-10);

        optimiser.optimize(parameters,
                (Tensor[] p) -> {
                    DenseVector x = (DenseVector) p[0];
                    double x1 = x.get(0);
                    double x2 = x.get(1);

                    double t1 = a - x1;
                    double t2 = x2 - x1 * x1;
                    double loss = (t1 * t1) + (b * t2 * t2);

                    // Gradient of Rosenbrock.
                    // df/dx = -2(a-x) - 4b x (y-x^2)
                    // df/dy = 2b (y-x^2)
                    double g1 = (-2.0 * t1) + (-4.0 * b * x1 * t2);
                    double g2 = 2.0 * b * t2;

                    return new LBFGS.GradAndLoss(new Tensor[]{DenseVector.createDenseVector(new double[]{g1, g2})}, loss);
                },
                (Tensor[] p) -> {
                    DenseVector x = (DenseVector) p[0];
                    double x1 = x.get(0);
                    double x2 = x.get(1);
                    double t1 = a - x1;
                    double t2 = x2 - x1 * x1;
                    return (t1 * t1) + (b * t2 * t2);
                });

        DenseVector finalX = (DenseVector) parameters.get()[0];

        assertEquals(1.0, finalX.get(0), 1e-3);
        assertEquals(1.0, finalX.get(1), 1e-3);

        // Ensure loss is close to optimum.
        double fx = (a - finalX.get(0));
        double fy = (finalX.get(1) - finalX.get(0) * finalX.get(0));
        double finalLoss = (fx * fx) + (b * fy * fy);
        assertTrue(finalLoss < 1e-9, "Expected loss < 1e-9, got " + finalLoss);

        // Ensure gradient norm is small.
        double g1 = (-2.0 * fx) + (-4.0 * b * finalX.get(0) * fy);
        double g2 = 2.0 * b * fy;
        double gradNorm = Math.sqrt((g1 * g1) + (g2 * g2));
        assertTrue(gradNorm < 1e-5, "Expected grad norm < 1e-5, got " + gradNorm);
    }

    /**
     * Minimal {@link Parameters} implementation for a single vector.
     */
    private static final class VectorParameters implements Parameters {
        private static final long serialVersionUID = 1L;

        private final Tensor[] params;

        private VectorParameters(DenseVector initial) {
            this.params = new Tensor[]{initial};
        }

        @Override
        public Tensor[] getEmptyCopy() {
            DenseVector x = (DenseVector) params[0];
            return new Tensor[]{new DenseVector(x.size())};
        }

        @Override
        public Tensor[] get() {
            return params;
        }

        @Override
        public void set(Tensor[] newWeights) {
            throw new UnsupportedOperationException("Not required for this test");
        }

        @Override
        public void update(Tensor[] gradients) {
            // LBFGS.update passes in the step vector, so we add it here.
            params[0].intersectAndAddInPlace(gradients[0]);
        }

        @Override
        public Tensor[] merge(Tensor[][] gradients, int size) {
            throw new UnsupportedOperationException("Not required for this test");
        }

        @Override
        public org.tribuo.math.protos.ParametersProto serialize() {
            throw new UnsupportedOperationException("Not required for this test");
        }
    }
}
