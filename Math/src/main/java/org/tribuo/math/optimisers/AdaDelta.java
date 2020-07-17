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
import org.tribuo.math.la.Tensor;

/**
 * An implementation of the AdaDelta gradient optimiser.
 * <p>
 * Creates two copies of the parameters to store learning rates.
 * <p>
 * See:
 * <pre>
 * Zeiler, MD.
 * "ADADELTA: an Adaptive Learning Rate Method"
 * arXiv preprint arXiv:1212.5701.
 * </pre>
 */
public class AdaDelta implements StochasticGradientOptimiser {

    @Config(description="Momentum value.")
    private double rho = 0.95;

    @Config(description="Epsilon for numerical stability.")
    private double epsilon = 1e-6;

    private Tensor[] gradsSquared;
    private Tensor[] velocitySquared;

    /**
     * It's recommended to keep rho at 0.95.
     * @param rho The rho value.
     * @param epsilon The epsilon value.
     */
    public AdaDelta(double rho, double epsilon) {
        this.rho = rho;
        this.epsilon = epsilon;
    }

    /**
     * Keeps rho at 0.95, passes through epsilon.
     * @param epsilon The epsilon value.
     */
    public AdaDelta(double epsilon) {
        this(0.95,epsilon);
    }

    /**
     * Sets rho to 0.95 and epsilon to 1e-6.
     */
    public AdaDelta() {
        this(0.95,1e-6);
    }

    @Override
    public void initialise(Parameters parameters) {
        gradsSquared = parameters.getEmptyCopy();
        velocitySquared = parameters.getEmptyCopy();
    }

    @Override
    public Tensor[] step(Tensor[] updates, double weight) {
        for (int i = 0; i < updates.length; i++) {
            gradsSquared[i].scaleInPlace(rho);
            gradsSquared[i].intersectAndAddInPlace(updates[i],(double a) -> a * a * (1.0 - rho));
            updates[i].hadamardProductInPlace(velocitySquared[i],(double a) -> Math.sqrt(a + epsilon));
            updates[i].hadamardProductInPlace(gradsSquared[i],(double a) -> 1.0 / (Math.sqrt(a + epsilon)));
            velocitySquared[i].scaleInPlace(rho);
            velocitySquared[i].intersectAndAddInPlace(updates[i],(double a) -> a * a * (1.0 - rho));
        }

        return updates;
    }

    @Override
    public String toString() {
        return "AdaDelta(rho="+rho+",epsilon="+epsilon+")";
    }

    @Override
    public void reset() {
        gradsSquared = null;
        velocitySquared = null;
    }

    @Override
    public AdaDelta copy() {
        return new AdaDelta(rho,epsilon);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"StochasticGradientOptimiser");
    }
}
