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

import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.math.StochasticGradientOptimiser;

import java.util.logging.Logger;

/**
 * CLI options for configuring a gradient optimiser.
 */
public class GradientOptimiserOptions implements Options {
    private static final Logger logger = Logger.getLogger(GradientOptimiserOptions.class.getName());

    public enum StochasticGradientOptimiserType {
        ADADELTA,
        ADAGRAD,
        ADAGRADRDA,
        ADAM,
        PEGASOS,
        RMSPROP,
        CONSTANTSGD,
        LINEARSGD,
        SQRTSGD
    }

    @Option(longName = "sgo-type", usage = "Selects the gradient optimiser. Defaults to ADAGRAD.")
    private StochasticGradientOptimiserType optimiserType = StochasticGradientOptimiserType.ADAGRAD;

    @Option(longName = "sgo-learning-rate", usage = "Learning rate for AdaGrad, AdaGradRDA, Adam, Pegasos.")
    public double learningRate = 0.18;

    @Option(longName = "sgo-epsilon", usage = "Epsilon for AdaDelta, AdaGrad, AdaGradRDA, Adam.")
    public double epsilon = 0.066;

    @Option(longName = "sgo-rho", usage = "Rho for RMSProp, AdaDelta, SGD with Momentum.")
    public double rho = 0.95;

    @Option(longName = "sgo-lambda", usage = "Lambda for Pegasos.")
    public double lambda = 1e-2;

    @Option(longName="sgo-parameter-averaging",usage="Use parameter averaging.")
    public boolean paramAve = false;

    @Option(longName="sgo-momentum",usage="Use momentum in SGD.")
    public SGD.Momentum momentum = SGD.Momentum.NONE;

    /**
     * Gets the configured gradient optimiser.
     * @return The gradient optimiser.
     */
    public StochasticGradientOptimiser getOptimiser() {
        StochasticGradientOptimiser sgo;
        switch(optimiserType) {
            case ADADELTA:
                sgo = new AdaDelta(rho,epsilon);
                break;
            case ADAGRAD:
                sgo = new AdaGrad(learningRate, epsilon);
                break;
            case ADAGRADRDA:
                sgo = new AdaGradRDA(learningRate, epsilon);
                break;
            case ADAM:
                sgo = new Adam(learningRate,epsilon);
                break;
            case PEGASOS:
                sgo = new Pegasos(learningRate,lambda);
                break;
            case RMSPROP:
                sgo = new RMSProp(learningRate,rho);
                break;
            case CONSTANTSGD:
                sgo = SGD.getSimpleSGD(learningRate,rho,momentum);
                break;
            case LINEARSGD:
                sgo = SGD.getLinearDecaySGD(learningRate,rho,momentum);
                break;
            case SQRTSGD:
                sgo = SGD.getSqrtDecaySGD(learningRate,rho,momentum);
                break;
            default:
                throw new IllegalArgumentException("Unhandled StochasticGradientOptimiser type: "+optimiserType);
        }
        if (paramAve) {
            logger.info("Using parameter averaging");
            return new ParameterAveraging(sgo);
        } else {
            return sgo;
        }
    }

}
