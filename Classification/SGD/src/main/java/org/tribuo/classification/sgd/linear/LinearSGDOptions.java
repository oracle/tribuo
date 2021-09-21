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

package org.tribuo.classification.sgd.linear;

import com.oracle.labs.mlrg.olcut.config.ArgumentException;
import com.oracle.labs.mlrg.olcut.config.Option;
import org.tribuo.Trainer;
import org.tribuo.classification.ClassificationOptions;
import org.tribuo.classification.sgd.LabelObjective;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.math.optimisers.GradientOptimiserOptions;

import java.util.logging.Logger;

/**
 * CLI options for training a linear classifier.
 */
public class LinearSGDOptions implements ClassificationOptions<LinearSGDTrainer> {
    private static final Logger logger = Logger.getLogger(LinearSGDOptions.class.getName());

    /**
     * Available loss types.
     */
    public enum LossEnum {
        /**
         * Hinge loss.
         */
        HINGE,
        /**
         * Log loss, i.e., cross-entropy.
         */
        LOG
    }

    /**
     * The gradient descent optimiser options.
     */
    public GradientOptimiserOptions sgoOptions;

    /**
     * Number of SGD epochs. Defaults to 5.
     */
    @Option(longName = "sgd-epochs", usage = "Number of SGD epochs. Defaults to 5.")
    public int sgdEpochs = 5;
    /**
     * Loss function. Defaults to LOG.
     */
    @Option(longName = "sgd-objective", usage = "Loss function. Defaults to LOG.")
    public LossEnum sgdObjective = LossEnum.LOG;
    /**
     * Log the objective after n examples. Defaults to 100.
     */
    @Option(longName = "sgd-logging-interval", usage = "Log the objective after <int> examples. Defaults to 100.")
    public int sgdLoggingInterval = 100;
    /**
     * Minibatch size. Defaults to 1.
     */
    @Option(longName = "sgd-minibatch-size", usage = "Minibatch size. Defaults to 1.")
    public int sgdMinibatchSize = 1;
    /**
     * Sets the random seed for the LinearSGDTrainer.
     */
    @Option(longName = "sgd-seed", usage = "Sets the random seed for the LinearSGDTrainer.")
    public long sgdSeed = Trainer.DEFAULT_SEED;

    /**
     * Returns the loss function specified in the arguments.
     *
     * @return The loss function.
     */
    public LabelObjective getLoss() {
        switch (sgdObjective) {
            case HINGE:
                return new Hinge();
            case LOG:
                return new LogMulticlass();
            default:
                throw new ArgumentException("sgd-objective", "Unknown loss function " + sgdObjective);
        }
    }

    @Override
    public LinearSGDTrainer getTrainer() {
        logger.info(String.format("Set logging interval to %d", sgdLoggingInterval));
        return new LinearSGDTrainer(getLoss(), sgoOptions.getOptimiser(), sgdEpochs, sgdLoggingInterval, sgdMinibatchSize, sgdSeed);
    }
}
