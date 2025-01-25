/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;

import java.util.function.Function;
import java.util.function.ToDoubleFunction;

/**
 * Implements the limited memory Broyden–Fletcher–Goldfarb–Shanno (BFGS) optimization algorithm for finding the minima
 * of a function using approximate second order gradient descent.
 * <p>
 * See:
 * <pre>
 * Nocedal, J. and Wright, S.
 * "Numerical Optimization (2nd Edition)"
 * Springer, 2006.
 * </pre>
 */
public final class LBFGS {

    /**
     * Return value from an L-BFGS evaluation.
     *
     * @param gradient The batch gradient.
     * @param loss     The batch loss.
     */
    public record GradAndLoss(Tensor[] gradient, double loss) { }

    private static final System.Logger logger = System.getLogger(LBFGS.class.getName());

    private final int maxIterations;

    private final double tolerance;

    private final double gradientTolerance;

    private final int memorySize;

    /**
     * Constructs a limited memory BFGS optimizer.
     *
     * @param memorySize        The memory size of the Hessian approximation, typically 10.
     * @param maxIterations     The maximum number of iterations, typically 1000.
     * @param tolerance         The convergence tolerance typically 1e-5.
     * @param gradientTolerance The zero gradient tolerance, typically 1e-4.
     */
    public LBFGS(int memorySize, int maxIterations, double tolerance, double gradientTolerance) {
        this.memorySize = memorySize;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.gradientTolerance = gradientTolerance;
    }

    /**
     * Constructs a limited memory BFGS optimizer with a memory size of 10, max iterations of 1000 and tolerance
     * of 1e-5.
     */
    public LBFGS() {
        this(10, 1000, 1e-5, 1e-4);
    }

    public void optimize(Parameters parameters, Function<Tensor[], GradAndLoss> lossAndGrad, ToDoubleFunction<Tensor[]> loss) {
        Tensor[] params = parameters.get();

        int histSize = 0;
        // step history
        var sArr = new DenseVector[this.memorySize];
        // gradient history
        var yArr = new DenseVector[this.memorySize];
        // scale history
        var rhoArr = new double[this.memorySize];
        double gamma = -1.0;

        double oldLoss = 0.0;

        LBFGS.GradAndLoss gradLoss = lossAndGrad.apply(params);
        DenseVector q = ravelArray(gradLoss.gradient);
        q.scaleInPlace(1.0 / q.twoNorm());
        DenseVector gradCopy = q.copy();
        DenseVector oldGrad;
        boolean converged = false;
        double lossValue = gradLoss.loss();
        for (int i = 0; i < maxIterations; i++) {
            // compute descent direction

            // - compute Hessian approximation
            // -- first recursion
            double[] alpha = new double[histSize];
            for (int j = 0; j < histSize; j++) {
                double curAlpha = rhoArr[j] * sArr[j].dot(q);
                alpha[j] = curAlpha;
                q.intersectAndAddInPlace(yArr[j], (double a) -> -curAlpha*a);
            }
            // -- compute initial approximation
            var r = q.scale(gamma);
            // -- second recursion
            for (int j = histSize-1; j >= 0; j--) {
                double beta = rhoArr[j] * yArr[j].dot(r);
                double curAlpha = alpha[j];
                r.intersectAndAddInPlace(sArr[j], (double a) -> a * (curAlpha - beta));
            }

            // reverse direction for gradient descent
            r.scaleInPlace(-1.0);

            // line search
            var stepSize = backtrackingLineSearch(loss, params, r, gradCopy);
            double gradNorm = r.twoNorm();
            r.scaleInPlace(stepSize);

            // check convergence
            if (Math.abs(lossValue - oldLoss) * 2 < (tolerance * (Math.abs(lossValue) + Math.abs(oldLoss) + 1e-10))) {
                converged = true;
                logger.log(System.Logger.Level.INFO, "L-BFGS converged at iteration " + i + " with loss value " + lossValue);
                break;
            } else if (gradNorm < gradientTolerance) {
                converged = true;
                logger.log(System.Logger.Level.INFO, "L-BFGS converged at iteration " + i + " with loss value " + lossValue + " due to minimum gradient");
                break;
            } else if (Double.isNaN(gradNorm) || Double.isInfinite(gradNorm)) {
                converged = true;
                logger.log(System.Logger.Level.WARNING, "L-BFGS diverged at iteration " + i + " with loss value " + lossValue);
                break;
            }

            // update parameters
            Tensor[] unravel = unravelVector(gradLoss.gradient, r);
            parameters.update(unravel);

            // update l-bfgs memory & compute gamma
            oldLoss = lossValue;
            // move old entries
            System.arraycopy(sArr, 0, sArr, 1, this.memorySize-1);
            System.arraycopy(yArr, 0, yArr, 1, this.memorySize-1);
            System.arraycopy(rhoArr, 0, rhoArr, 1, this.memorySize-1);
            if (histSize < this.memorySize) {
                histSize++;
            }
            // write new cache entries
            sArr[0] = r;
            // compute new gradient for position i+1
            gradLoss = lossAndGrad.apply(params);
            lossValue = gradLoss.loss();
            q = ravelArray(gradLoss.gradient);
            oldGrad = gradCopy;
            gradCopy = q.copy();
            yArr[0] = gradCopy.subtract(oldGrad);
            double sdoty = sArr[0].dot(yArr[0]);
            rhoArr[0] = 1.0 / sdoty;
            gamma = sdoty / (yArr[0].dot(yArr[0]));
            logger.log(System.Logger.Level.INFO, "L-BGFS iteration " + i + ", loss = " + lossValue + " gradNorm " + gradNorm + " gamma " + gamma);
        }
        if (!converged) {
            logger.log(System.Logger.Level.INFO, "Max iterations exceeded at loss " + lossValue);
        }
    }

    /**
     * Ravels the tensor array into a single {@link DenseVector}.
     * @param gradient The tensor array.
     * @return The ravelled and stacked tensor.
     */
    private static DenseVector ravelArray(Tensor[] gradient) {
        DenseVector[] arr = new DenseVector[gradient.length];
        int i = 0;
        int numElements = 0;
        for (var g : gradient) {
            if (g instanceof DenseVector gVec) {
                arr[i] = gVec;
            } else if (g instanceof DenseMatrix gMat) {
                arr[i] = gMat.ravel();
            } else if (g instanceof SparseVector gVec) {
                arr[i] = gVec.densify();
            } else if (g instanceof DenseSparseMatrix gMat) {
                arr[i] = gMat.densify().ravel();
            } else {
                throw new IllegalArgumentException("Unexpected tensor type " + g.getClass());
            }
            numElements += arr[i].size();
            i++;
        }
        DenseVector output = new DenseVector(numElements);
        int curPos = 0;
        for (var g : arr) {
            output.setElements(g, curPos, 0, g.size());
            curPos += g.size();
        }
        if (numElements != curPos) {
            throw new IllegalStateException("Lost some values somewhere, curPos " + curPos + ", numElements " + numElements);
        }
        return output;
    }

    /**
     * Unravels the vector into a tensor array.
     * @param oldGrad The old tensor array.
     * @param gradient The ravelled vector.
     * @return The unravelled tensor array.
     */
    private static Tensor[] unravelVector(Tensor[] oldGrad, DenseVector gradient) {
        Tensor[] output = new Tensor[oldGrad.length];

        int curPos = 0;
        for (int i = 0; i < oldGrad.length; i++) {
            if (oldGrad[i] instanceof SGDVector g) {
                var newOutput = new DenseVector(g.size());
                newOutput.setElements(gradient, 0, curPos, newOutput.size());
                output[i] = newOutput;
                curPos += newOutput.size();
            } else if (oldGrad[i] instanceof Matrix g) {
                var newOutput = new DenseMatrix(g.getDimension1Size(), g.getDimension2Size());
                newOutput.set(gradient, curPos);
                output[i] = newOutput;
                curPos += newOutput.getDimension1Size() * newOutput.getDimension2Size();
            } else {
                throw new IllegalArgumentException("Unexpected tensor type " + oldGrad[i].getClass());
            }
        }

        return output;
    }

    // Constants from Nocedal & Wright 2006, Numerical Optimization (2nd Edition)
    private static final double C_ONE = 1e-4;
    private static final double C_TWO = 0.9;
    private static final int MAX_LINESEARCH_ITR = 50;

    private static double backtrackingLineSearch(ToDoubleFunction<Tensor[]> lossFunc, Tensor[] params,  DenseVector descentDirection, DenseVector gradient) {
        final double ALPHA_INIT = 1.0d;
        var alpha = ALPHA_INIT;

        var startLoss = lossFunc.applyAsDouble(params);
        DenseVector raveledParams = ravelArray(params);
        raveledParams.intersectAndAddInPlace(descentDirection, a -> a * ALPHA_INIT);
        Tensor[] newPos = unravelVector(params, raveledParams);
        double curLoss = lossFunc.applyAsDouble(newPos);
        var convergenceLimit = C_ONE * descentDirection.dot(gradient);
        /*
        System.out.println("Descent direction " + descentDirection);
        System.out.println("Gradient " + gradient);
        System.out.println("Params " + params[0]);
        System.out.println("New pos " + newPos[0]);
        if (convergenceLimit >= 0) {
            throw new IllegalStateException("Invalid convergence limit, descent direction & gradient point in orthogonal or opposite directions, convergenceLimit " + convergenceLimit);
        }
         */
        int itr = 0;
        while ((curLoss > (startLoss + alpha*convergenceLimit)) && (itr < MAX_LINESEARCH_ITR)) {
            double newAlpha = alpha * C_TWO;
            double alphaDiff = newAlpha - alpha;
            alpha = newAlpha;
            raveledParams.intersectAndAddInPlace(descentDirection, a -> a * alphaDiff);
            newPos = unravelVector(params, raveledParams);
            curLoss = lossFunc.applyAsDouble(newPos);
            itr++;
        }
        if (itr == MAX_LINESEARCH_ITR) {
            logger.log(System.Logger.Level.INFO, "Exceeded line search iterations with alpha " + alpha + ", convergence limit " + convergenceLimit + ", start loss " + startLoss + ", end loss " + curLoss);
        } else {
            logger.log(System.Logger.Level.INFO, "Line search terminated with alpha " + alpha + " at itr " + itr + " with end loss " + curLoss + ", start loss " + startLoss + ", convergence limit " + convergenceLimit);
        }

        return alpha;
    }
}
