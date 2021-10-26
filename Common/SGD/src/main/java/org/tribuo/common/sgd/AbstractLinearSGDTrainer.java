/*
 * Copyright (c) 2020, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.Output;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.StochasticGradientOptimiser;

import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * A trainer for a linear model which uses SGD.
 * <p>
 * It's an {@link AbstractSGDTrainer} operating on {@link LinearParameters}, with
 * the bias folded into the features.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public abstract class AbstractLinearSGDTrainer<T extends Output<T>,U,V extends AbstractLinearSGDModel<T>> extends AbstractSGDTrainer<T,U,V,LinearParameters> {
    private static final Logger logger = Logger.getLogger(AbstractLinearSGDTrainer.class.getName());

    /**
     * Constructs an SGD trainer for a linear model.
     * @param optimiser The gradient optimiser to use.
     * @param epochs The number of epochs (complete passes through the training data).
     * @param loggingInterval Log the loss after this many iterations. If -1 don't log anything.
     * @param minibatchSize The size of any minibatches.
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    protected AbstractLinearSGDTrainer(StochasticGradientOptimiser optimiser, int epochs, int loggingInterval, int minibatchSize, long seed) {
        super(optimiser,epochs,loggingInterval,minibatchSize,seed,true);
        postConfig();
    }

    /**
     * For olcut.
     */
    protected AbstractLinearSGDTrainer() {
        super(true);
    }

    /**
     * Returns the default model name.
     * @return The default model name.
     */
    @Override
    protected String getName() {
        return "linear-sgd-model";
    }

    /**
     * Constructs the trainable parameters object, in this case a {@link LinearParameters} containing
     * a single weight matrix.
     * @param numFeatures The number of input features.
     * @param numOutputs The number of output dimensions.
     * @param localRNG The RNG to use for parameter initialisation.
     * @return The trainable parameters.
     */
    @Override
    protected LinearParameters createParameters(int numFeatures, int numOutputs, SplittableRandom localRNG) {
        return new LinearParameters(numFeatures+1,numOutputs);
    }

}
