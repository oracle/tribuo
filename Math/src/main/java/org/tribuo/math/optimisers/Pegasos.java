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
import org.tribuo.math.la.Tensor;
import org.tribuo.math.optimisers.util.ShrinkingMatrix;
import org.tribuo.math.optimisers.util.ShrinkingTensor;
import org.tribuo.math.optimisers.util.ShrinkingVector;

/**
 * An implementation of the Pegasos gradient optimiser used primarily for solving the SVM problem.
 * <p>
 * This gradient optimiser rewrites all the {@link Tensor}s in the {@link Parameters}
 * with {@link ShrinkingTensor}. This means it keeps a different value in the {@link Tensor}
 * to the one produced when you call get(), so it can correctly apply regularisation to the parameters.
 * When {@link Pegasos#finalise()} is called it rewrites the {@link Parameters} with standard dense {@link Tensor}s.
 * Follows the implementation in Factorie.
 * <p>
 * Pegasos is remarkably touchy about it's learning rates. The defaults work on a couple of examples, but it
 * requires tuning to work properly on a specific dataset.
 * <p>
 * See:
 * <pre>
 * Shalev-Shwartz S, Singer Y, Srebro N, Cotter A
 * "Pegasos: Primal Estimated Sub-Gradient Solver for SVM"
 * Mathematical Programming, 2011.
 * </pre>
 */
public class Pegasos implements StochasticGradientOptimiser {

    @Config(description="Step size shrinkage.")
    private double lambda = 1e-2;

    @Config(description="Base learning rate.")
    private double baseRate = 0.1;

    private int iteration = 1;
    private Parameters parameters;

    /**
     * Added for olcut configuration.
     */
    private Pegasos() { }

    /**
     * Constructs a Pegasos optimiser with the specified parameters.
     * @param baseRate The base learning rate.
     * @param lambda The regularisation parameter.
     */
    public Pegasos(double baseRate, double lambda) {
        this.baseRate = baseRate;
        this.lambda = lambda;
    }

    @Override
    public void initialise(Parameters parameters) {
        this.parameters = parameters;
        Tensor[] curParams = parameters.get();
        Tensor[] newParams = new Tensor[curParams.length];
        for (int i = 0; i < newParams.length; i++) {
            if (curParams[i] instanceof DenseVector) {
                newParams[i] = new ShrinkingVector(((DenseVector) curParams[i]), baseRate, lambda);
            } else if (curParams[i] instanceof DenseMatrix) {
                newParams[i] = new ShrinkingMatrix(((DenseMatrix) curParams[i]), baseRate, lambda);
            } else {
                throw new IllegalStateException("Unknown Tensor subclass");
            }
        }
        parameters.set(newParams);
    }

    @Override
    public Tensor[] step(Tensor[] updates, double weight) {
        double eta_t = baseRate / (lambda * iteration);
        for (Tensor t : updates) {
            t.scaleInPlace(eta_t * weight);
        }
        iteration++;
        return updates;
    }

    @Override
    public String toString() {
        return "Pegasos(baseRate=" + baseRate + ",lambda=" + lambda + ")";
    }

    @Override
    public void finalise() {
        Tensor[] curParams = parameters.get();
        Tensor[] newParams = new Tensor[curParams.length];
        for (int i = 0; i < newParams.length; i++) {
            if (curParams[i] instanceof ShrinkingTensor) {
                newParams[i] = ((ShrinkingTensor) curParams[i]).convertToDense();
            } else {
                throw new IllegalStateException("Finalising a Parameters which wasn't initialised with Pegasos");
            }
        }
        parameters.set(newParams);
    }

    @Override
    public void reset() {
        parameters = null;
        iteration = 1;
    }

    @Override
    public Pegasos copy() {
        return new Pegasos(lambda,baseRate);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"StochasticGradientOptimiser");
    }
}

