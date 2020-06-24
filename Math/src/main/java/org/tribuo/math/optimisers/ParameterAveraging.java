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
 * Averages the parameters across a gradient run.
 * <p>
 * Wraps an inner gradient optimiser. Only changes the values when {@link ParameterAveraging#finalise()} is called
 * <p>
 * See:
 * <pre>
 * Polyak BT, Juditsky AB
 * "Acceleration of Stochastic Approximation by Averaging"
 * SIAM Journal on Control and Optimization, 1992.
 * </pre>
 */
public class ParameterAveraging implements StochasticGradientOptimiser {

    @Config(mandatory = true,description="Inner optimiser to average parameters across.")
    private StochasticGradientOptimiser optimiser;

    private int iterations = 0;
    private Tensor[] weights;
    private Parameters parameters;

    /**
     * Adds parameter averaging around a gradient optimiser.
     * @param optimiser The inner optimiser to use to scale the gradients.
     */
    public ParameterAveraging(StochasticGradientOptimiser optimiser) {
        this.optimiser = optimiser;
    }

    /**
     * For olcut.
     */
    private ParameterAveraging() { }

    @Override
    public void initialise(Parameters parameters) {
        optimiser.initialise(parameters);
        weights = parameters.getEmptyCopy();
        this.parameters = parameters;
    }

    /**
     * This passes the gradient update to the inner optimiser, then updates
     * the average weight values.
     * @param updates An array of gradients.
     * @param weight The weight for the current gradients.
     * @return The gradients from the inner optimiser.
     */
    @Override
    public Tensor[] step(Tensor[] updates, double weight) {
        iterations++;
        Tensor[] output = optimiser.step(updates, weight);
        for (int i = 0; i < output.length; i++) {
            weights[i].intersectAndAddInPlace(output[i],(double a) -> a * iterations);
        }
        return output;
    }

    /**
     * This sets the parameters to their average value.
     */
    @Override
    public void finalise() {
        Tensor[] tmp = parameters.get();
        for (int i = 0; i < tmp.length; i++) {
            tmp[i].intersectAndAddInPlace(weights[i],(double a) -> -a / iterations);
        }
    }

    @Override
    public String toString() {
        return "ParameterAveraging(optimiser="+optimiser.toString()+")";
    }

    @Override
    public void reset() {
        optimiser.reset();
        iterations = 0;
        weights = null;
    }

    @Override
    public ParameterAveraging copy() {
        return new ParameterAveraging(optimiser.copy());
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"StochasticGradientOptimiser");
    }
}
