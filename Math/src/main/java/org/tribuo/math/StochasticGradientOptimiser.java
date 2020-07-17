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

package org.tribuo.math;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.optimisers.ParameterAveraging;

/**
 * Interface for gradient based optimisation methods.
 * <p>
 * Order of use:
 * <ul>
 * <li>{@link StochasticGradientOptimiser#initialise(Parameters)}</li>
 * <li>take many {@link StochasticGradientOptimiser#step(Tensor[], double)}s</li>
 * <li>{@link StochasticGradientOptimiser#finalise()}</li>
 * <li>{@link StochasticGradientOptimiser#reset()}</li>
 * </ul>
 *
 * Deviating from this order will cause unexpected behaviour.
 */
public interface StochasticGradientOptimiser extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Initialises the gradient optimiser.
     * <p>
     * Configures any learning rate parameters.
     * @param parameters The parameters to optimise.
     */
    default public void initialise(Parameters parameters) {}

    /**
     * Take a {@link Tensor} array of gradients and transform them
     * according to the current weight and learning rates.
     * <p>
     * Can return the same {@link Tensor} array or a new one.
     * @param updates An array of gradients.
     * @param weight The weight for the current gradients.
     * @return A {@link Tensor} array of gradients.
     */
    public Tensor[] step(Tensor[] updates, double weight);

    /**
     * Finalises the gradient optimisation, setting the parameters to their correct values.
     * Used for {@link ParameterAveraging} amongst others.
     */
    default public void finalise() {}

    /**
     * Resets the optimiser so it's ready to optimise a new {@link Parameters}.
     */
    public void reset();

    /**
     * Copies a gradient optimiser with it's configuration. Usually calls the copy constructor.
     * @return A gradient optimiser with the same configuration, but independent state.
     */
    public StochasticGradientOptimiser copy();
}
