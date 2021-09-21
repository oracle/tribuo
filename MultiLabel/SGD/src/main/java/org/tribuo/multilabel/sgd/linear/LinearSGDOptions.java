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

package org.tribuo.multilabel.sgd.linear;

import com.oracle.labs.mlrg.olcut.config.ArgumentException;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.Trainer;
import org.tribuo.multilabel.sgd.MultiLabelObjective;
import org.tribuo.multilabel.sgd.objectives.Hinge;
import org.tribuo.multilabel.sgd.objectives.BinaryCrossEntropy;
import org.tribuo.math.optimisers.GradientOptimiserOptions;

import java.util.logging.Logger;

/**
 * CLI options for training a linear classifier.
 */
public class LinearSGDOptions implements Options {
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
         * Log loss, i.e., binary cross-entropy.
         */
        SIGMOID
    }

    /**
     * The gradient descent optimiser options.
     */
    public GradientOptimiserOptions sgoOptions;

    /**
     * Number of SGD epochs.
     */
    @Option(longName = "sgd-epochs", usage = "Number of SGD epochs.")
    public int sgdEpochs = 5;
    /**
     * Loss function.
     */
    @Option(longName = "sgd-objective", usage = "Loss function.")
    public LossEnum sgdObjective = LossEnum.SIGMOID;
    /**
     * Log the objective after n examples.
     */
    @Option(longName = "sgd-logging-interval", usage = "Log the objective after <int> examples.")
    public int sgdLoggingInterval = 100;
    /**
     * Minibatch size.
     */
    @Option(longName = "sgd-minibatch-size", usage = "Minibatch size.")
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
    public MultiLabelObjective getLoss() {
        switch (sgdObjective) {
            case HINGE:
                return new Hinge();
            case SIGMOID:
                return new BinaryCrossEntropy();
            default:
                throw new ArgumentException("sgd-objective", "Unknown loss function " + sgdObjective);
        }
    }

    /**
     * Gets the LinearSGDTrainer specified by the options in this object.
     * @return The configured trainer.
     */
    public LinearSGDTrainer getTrainer() {
        logger.info(String.format("Set logging interval to %d", sgdLoggingInterval));
        return new LinearSGDTrainer(getLoss(), sgoOptions.getOptimiser(), sgdEpochs, sgdLoggingInterval, sgdMinibatchSize, sgdSeed);
    }
}
