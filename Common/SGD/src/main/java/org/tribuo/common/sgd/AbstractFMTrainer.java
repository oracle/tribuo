/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.sgd;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.tribuo.Output;
import org.tribuo.math.StochasticGradientOptimiser;

import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * A trainer for a quadratic factorization machine model which uses SGD.
 * <p>
 * It's an {@link AbstractSGDTrainer} operating on {@link FMParameters}.
 * <p>
 * See:
 * <pre>
 * Rendle, S.
 * Factorization machines.
 * 2010 IEEE International Conference on Data Mining
 * </pre>
 */
public abstract class AbstractFMTrainer<T extends Output<T>, U> extends AbstractSGDTrainer<T, U, AbstractFMModel<T>, FMParameters> {
    private static final Logger logger = Logger.getLogger(AbstractFMTrainer.class.getName());

    @Config(mandatory = true, description = "The size of the factorized feature representation.")
    protected int factorizedDimSize;

    @Config(mandatory = true, description = "The variance of the initializer.")
    protected double variance;

    /**
     * Constructs an SGD trainer for a factorization machine.
     *
     * @param optimiser         The gradient optimiser to use.
     * @param epochs            The number of epochs (complete passes through the training data).
     * @param loggingInterval   Log the loss after this many iterations. If -1 don't log anything.
     * @param minibatchSize     The size of any minibatches.
     * @param seed              A seed for the random number generator, used to shuffle the examples before each epoch.
     * @param factorizedDimSize Size of the factorized feature representation.
     * @param variance          The variance of the initializer.
     */
    protected AbstractFMTrainer(StochasticGradientOptimiser optimiser, int epochs, int loggingInterval,
                                int minibatchSize, long seed, int factorizedDimSize, double variance) {
        super(optimiser, epochs, loggingInterval, minibatchSize, seed, false);
        this.factorizedDimSize = factorizedDimSize;
        this.variance = variance;
        postConfig();
    }

    /**
     * For olcut.
     */
    protected AbstractFMTrainer() {
        super(false);
    }

    @Override
    public void postConfig() {
        super.postConfig();
        if (factorizedDimSize < 1) {
            throw new PropertyException("", "factorizedDimSize", "Value must be positive.");
        }
        if (variance <= 0.0) {
            throw new PropertyException("", "variance", "Value must be positive.");
        }
    }

    /**
     * Returns the default model name.
     *
     * @return The default model name.
     */
    @Override
    protected String getName() {
        return "factorization-machine-model";
    }

    /**
     * Constructs the trainable parameters object, in this case a {@link FMParameters} containing
     * a weight matrix for the feature weights and a series of weight matrices for the factorized
     * feature representation.
     *
     * @param numFeatures The number of input features.
     * @param numOutputs  The number of output dimensions.
     * @param localRNG    The RNG to use for parameter initialisation.
     * @return The trainable parameters.
     */
    @Override
    protected FMParameters createParameters(int numFeatures, int numOutputs, SplittableRandom localRNG) {
        return new FMParameters(localRNG, numFeatures, numOutputs, factorizedDimSize, variance);
    }

}
