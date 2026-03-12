/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;

import java.util.Arrays;
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
public final class LBFGS implements Configurable, Provenancable<ConfiguredObjectProvenance> {
    private static final System.Logger logger = System.getLogger(LBFGS.class.getName());

    /**
     * Return value from an L-BFGS evaluation.
     *
     * @param gradient The batch gradient.
     * @param loss     The batch loss.
     */
    public record GradAndLoss(Tensor[] gradient, double loss) { }

    @Config(description = "The maximum number of iterations.")
    private int maxIterations = 1000;

    @Config(description = "The loss convergence tolerance.")
    private double tolerance = 1e-5;

    @Config(description = "The gradient convergence tolerance.")
    private double gradientTolerance = 1e-4;

    @Config(description = "The number of previous gradients to keep in the Hessian approximation.")
    private int memorySize = 10;

    @Config(description = "Use a Wolfe line search, instead of a simpler backtracking line search.")
    private boolean useWolfe = true;

    /**
     * Constructs a limited memory BFGS optimizer with a Wolfe line search.
     *
     * @param memorySize        The memory size of the Hessian approximation, typically 10.
     * @param maxIterations     The maximum number of iterations, typically 1000.
     * @param tolerance         The convergence tolerance typically 1e-5.
     * @param gradientTolerance The zero gradient tolerance, typically 1e-4.
     */
    public LBFGS(int memorySize, int maxIterations, double tolerance, double gradientTolerance) {
        this(memorySize, maxIterations, tolerance, gradientTolerance, true);
    }

    /**
     * Constructs a limited memory BFGS optimizer.
     *
     * @param memorySize        The memory size of the Hessian approximation, typically 10.
     * @param maxIterations     The maximum number of iterations, typically 1000.
     * @param tolerance         The convergence tolerance typically 1e-5.
     * @param gradientTolerance The zero gradient tolerance, typically 1e-4.
     * @param useWolfe          Use a Wolfe line search if true, otherwise use a backtracking one.
     */
    public LBFGS(int memorySize, int maxIterations, double tolerance, double gradientTolerance, boolean useWolfe) {
        this.memorySize = memorySize;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.gradientTolerance = gradientTolerance;
        this.useWolfe = useWolfe;
    }

    /**
     * Constructs a limited memory BFGS optimizer with a memory size of 10, max iterations of 1000, tolerance
     * of 1e-5 and a Wolfe line search.
     */
    public LBFGS() {
        this(10, 1000, 1e-5, 1e-4);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "LBFGS");
    }

    /**
     * Minimizes the supplied function using limited memory BFGS.
     * @param parameters The function parameters.
     * @param lossAndGrad A function which generates the loss and gradient given the parameters.
     * @param loss A function which generates the loss given the parameters.
     */
    public void optimize(Parameters parameters, Function<Tensor[], GradAndLoss> lossAndGrad, ToDoubleFunction<Tensor[]> loss) {
        Tensor[] params = parameters.get();

        int histSize = 0;
        // step history
        var sArr = new DenseVector[this.memorySize];
        // gradient history
        var yArr = new DenseVector[this.memorySize];
        // scale history
        var rhoArr = new double[this.memorySize];
        double[] alpha = new double[this.memorySize];
        double gamma = 1.0;

        double oldLoss = 0.0;

        LBFGS.GradAndLoss gradLoss = lossAndGrad.apply(params);
        DenseVector q = ravelArray(gradLoss.gradient);
        DenseVector gradCopy = q.copy();
        DenseVector oldGrad;
        boolean converged = false;
        double lossValue = gradLoss.loss();
        for (int i = 0; i < maxIterations; i++) {
            // Zero the alphas from the previous iteration.
            Arrays.fill(alpha, 0.0);

            // compute descent direction
            // - compute Hessian approximation
            // -- first recursion
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

            // Defensive check: if the approximate inverse-Hessian produced a non-descent direction,
            // fall back to steepest descent. Wolfe line search requires g^T p < 0.
            double gradDotDescent = gradCopy.dot(r);
            if (gradDotDescent > 0.0) {
                logger.log(System.Logger.Level.WARNING,
                        "L-BFGS produced a non-descent direction (g^T p=" + gradDotDescent + ") at iteration " + i + ", falling back to steepest descent.");
                r = gradCopy.copy();
                r.scaleInPlace(-1.0);
            }

            // line search
            double stepSize;
            if (this.useWolfe) {
                stepSize = wolfeLineSearch(lossAndGrad, loss, params, r, gradCopy);
            } else {
                stepSize = backtrackingLineSearch(loss, params, r, gradCopy);
            }
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
                logger.log(System.Logger.Level.WARNING, "L-BFGS diverged at iteration " + i + " with loss value " + lossValue + " due to a NaN gradient");
                break;
            } else if (stepSize == 0.0) {
                converged = true;
                logger.log(System.Logger.Level.WARNING, "L-BFGS line search could not proceed due to a diverging descent direction at iteration " + i + " with loss value " + lossValue);
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
            // compute new gradient for position i+1
            gradLoss = lossAndGrad.apply(params);
            lossValue = gradLoss.loss();
            q = ravelArray(gradLoss.gradient);
            oldGrad = gradCopy;
            gradCopy = q.copy();

            // s_k = x_{k+1} - x_k as the parameter update is r.
            sArr[0] = r;
            yArr[0] = gradCopy.subtract(oldGrad);
            double ydoty = yArr[0].dot(yArr[0]);
            double sdoty = sArr[0].dot(yArr[0]);

            // Skip invalid curvature updates. Standard L-BFGS requires sdoty > 0.
            // If invalid, set rho[0] to 0 so this history element is ignored.
            if (sdoty > 1e-16 && ydoty > 0.0) {
                rhoArr[0] = 1.0 / sdoty;
                gamma = sdoty / ydoty;
            } else {
                rhoArr[0] = 0.0;
                // keep gamma unchanged
                logger.log(System.Logger.Level.WARNING,
                        "Invalid curvature update at iteration " + i + " (sdoty=" + sdoty + ", ydoty=" + ydoty + "), skipping L-BFGS memory update.");
            }
            if (ydoty == 0.0) {
                logger.log(System.Logger.Level.INFO, "L-BFGS converged as gradient is unchanging");
                break;
            }
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
        if (convergenceLimit > 0) {
            logger.log(System.Logger.Level.WARNING, "Invalid convergence limit, descent direction & gradient point in orthogonal or opposite directions, convergenceLimit " + convergenceLimit);
            return 0.0;
        }
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

    /**
     * A strong Wolfe line search.
     * <p>
     * Enforces:
     * <ul>
     *     <li><b>Armijo</b>: {@code f(x + alpha p) <= f(x) + c1 * alpha * g(x)^T p}</li>
     *     <li><b>Strong curvature</b>: {@code |g(x + alpha p)^T p| <= c2 * |g(x)^T p|}</li>
     * </ul>
     * <p>
     * This is a bracketing + zoom line search as described in Nocedal & Wright (2006).
     * <p>
     * Requirements:
     * <ul>
     *     <li>{@code gradient} is the gradient at {@code params}.</li>
     *     <li>{@code descentDirection} is a descent direction: {@code gradient.dot(descentDirection) < 0}.</li>
     * </ul>
     *
     * @param lossAndGrad A function returning loss and gradient at a position.
     * @param lossFunc A function returning loss at a position. Used to reduce gradient evaluations.
     * @param params The current parameter tensors, not mutated by the search.
     * @param descentDirection The search direction p.
     * @param gradient The gradient at position {@code params}.
     * @return A step size alpha which (approximately) satisfies the strong Wolfe conditions.
     */
    private static double wolfeLineSearch(Function<Tensor[], GradAndLoss> lossAndGrad,
                                         ToDoubleFunction<Tensor[]> lossFunc,
                                         Tensor[] params,
                                         DenseVector descentDirection,
                                         DenseVector gradient) {
        final double alphaInit = 1.0;
        final double alphaMax = 50.0;

        final double phi0 = lossFunc.applyAsDouble(params);
        final double dphi0 = gradient.dot(descentDirection);
        if (!(dphi0 < 0.0)) {
            throw new IllegalStateException("wolfeLineSearch requires a descent direction, gradient dot direction=" + dphi0);
        }

        double alphaPrev = 0.0;
        double phiPrev = phi0;
        double alpha = alphaInit;

        for (int itr = 0; itr < MAX_LINESEARCH_ITR; itr++) {
            double phi = phiAt(lossFunc, params, descentDirection, alpha);

            if ((phi > phi0 + C_ONE * alpha * dphi0) || (itr > 0 && phi >= phiPrev)) {
                return zoom(lossAndGrad, lossFunc, params, descentDirection, phi0, dphi0, alphaPrev, alpha, C_ONE, C_TWO);
            }

            double dphi = dPhiAt(lossAndGrad, params, descentDirection, alpha);
            if (Math.abs(dphi) <= C_TWO * Math.abs(dphi0)) {
                logger.log(System.Logger.Level.INFO, "Line search terminated with alpha " + alpha + " at itr " + itr + " with end loss " + phi + ", start loss " + phi0 + ", convergence limit " + C_TWO*Math.abs(dphi0));
                return alpha;
            }

            if (dphi >= 0.0) {
                return zoom(lossAndGrad, lossFunc, params, descentDirection, phi0, dphi0, alpha, alphaPrev, C_ONE, C_TWO);
            }

            alphaPrev = alpha;
            phiPrev = phi;
            alpha = Math.min(alpha * 2.0, alphaMax);
        }

        logger.log(System.Logger.Level.INFO, "Exceeded line search iterations with alpha " + alpha + ", convergence limit " + C_TWO*Math.abs(dphi0) + ", start loss " + phi0 + ", end loss " + phiPrev);
        return alpha;
    }

    private static double zoom(Function<Tensor[], GradAndLoss> lossAndGrad,
                               ToDoubleFunction<Tensor[]> lossFunc,
                               Tensor[] params,
                               DenseVector p,
                               double phi0,
                               double dphi0,
                               double aLo,
                               double aHi,
                               double c1,
                               double c2) {
        double phiLo = phiAt(lossFunc, params, p, aLo);

        for (int itr = 0; itr < MAX_LINESEARCH_ITR; itr++) {
            // Bisect range
            double aJ = 0.5 * (aLo + aHi);
            double phiJ = phiAt(lossFunc, params, p, aJ);

            if ((phiJ > phi0 + c1 * aJ * dphi0) || (phiJ >= phiLo)) {
                aHi = aJ;
            } else {
                double dphiJ = dPhiAt(lossAndGrad, params, p, aJ);
                if (Math.abs(dphiJ) <= c2 * Math.abs(dphi0)) {
                    return aJ;
                }
                if (dphiJ * (aHi - aLo) >= 0.0) {
                    aHi = aLo;
                }
                aLo = aJ;
                phiLo = phiJ;
            }

            // Interval disappeared
            if (Math.abs(aHi - aLo) < 1e-12) {
                return aJ;
            }
        }

        return 0.5 * (aLo + aHi);
    }

    private static double phiAt(ToDoubleFunction<Tensor[]> lossFunc, Tensor[] params, DenseVector p, double alpha) {
        DenseVector raveledParams = ravelArray(params);
        raveledParams.intersectAndAddInPlace(p, a -> a * alpha);
        Tensor[] newPos = unravelVector(params, raveledParams);
        return lossFunc.applyAsDouble(newPos);
    }

    private static double dPhiAt(Function<Tensor[], GradAndLoss> lossAndGrad, Tensor[] params, DenseVector p, double alpha) {
        DenseVector raveledParams = ravelArray(params);
        raveledParams.intersectAndAddInPlace(p, a -> a * alpha);
        Tensor[] newPos = unravelVector(params, raveledParams);
        DenseVector g = ravelArray(lossAndGrad.apply(newPos).gradient());
        return g.dot(p);
    }
}
