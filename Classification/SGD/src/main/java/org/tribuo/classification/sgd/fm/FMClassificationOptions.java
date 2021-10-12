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

package org.tribuo.classification.sgd.fm;

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
 * CLI options for training a factorization machine classifier.
 */
public class FMClassificationOptions implements ClassificationOptions<FMClassificationTrainer> {
    private static final Logger logger = Logger.getLogger(FMClassificationOptions.class.getName());

    /**
     * Available loss types.
     */
    public enum LossEnum {
        /**
         * Hinge loss (like an SVM).
         */
        HINGE,
        /**
         * Log loss (i.e., a logistic regression).
         */
        LOG
    }

    public GradientOptimiserOptions sgoOptions;

    /**
     * Number of SGD epochs.
     */
    @Option(longName = "fm-epochs", usage = "Number of SGD epochs.")
    public int fmEpochs = 5;
    /**
     * Loss function.
     */
    @Option(longName = "fm-objective", usage = "Loss function.")
    public LossEnum fmObjective = LossEnum.LOG;
    /**
     * Log the objective after n examples.
     */
    @Option(longName = "fm-logging-interval", usage = "Log the objective after <int> examples.")
    public int fmLoggingInterval = 100;
    /**
     * Minibatch size.
     */
    @Option(longName = "fm-minibatch-size", usage = "Minibatch size.")
    public int fmMinibatchSize = 1;
    /**
     * Sets the random seed for the FMClassificationTrainer.
     */
    @Option(longName = "fm-seed", usage = "Sets the random seed for the FMClassificationTrainer.")
    private long fmSeed = Trainer.DEFAULT_SEED;
    /**
     * Factor size.
     */
    @Option(longName = "fm-factor-size", usage = "Factor size.")
    public int fmFactorSize = 5;
    /**
     * Variance of the initialization gaussian.
     */
    @Option(longName = "fm-variance", usage = "Variance of the initialization gaussian.")
    public double fmVariance = 0.5;

    /**
     * Returns the loss function specified in the arguments.
     * @return The loss function.
     */
    public LabelObjective getLoss() {
        switch (fmObjective) {
            case HINGE:
                return new Hinge();
            case LOG:
                return new LogMulticlass();
            default:
                throw new ArgumentException("sgd-objective", "Unknown loss function " + fmObjective);
        }
    }

    @Override
    public FMClassificationTrainer getTrainer() {
        logger.info(String.format("Set logging interval to %d", fmLoggingInterval));
        return new FMClassificationTrainer(getLoss(), sgoOptions.getOptimiser(), fmEpochs, fmLoggingInterval,
                fmMinibatchSize, fmSeed, fmFactorSize, fmVariance);
    }
}
