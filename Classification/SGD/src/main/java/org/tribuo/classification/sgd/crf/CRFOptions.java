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

package org.tribuo.classification.sgd.crf;

import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.Trainer;
import org.tribuo.math.optimisers.GradientOptimiserOptions;

/**
 * CLI options for training a linear chain CRF model.
 */
public class CRFOptions implements Options {

    /**
     * Options for the gradient optimiser.
     */
    public GradientOptimiserOptions sgoOptions;

    @Option(longName = "crf-epochs", usage = "Epochs of SGD.")
    private int epochs = 5;

    @Option(longName = "crf-logging-interval", usage = "Logging interval for loss function.")
    private int loggingInterval = 100;

    @Option(longName = "crf-seed", usage = "Sets the random seed for the CRF.")
    private long seed = Trainer.DEFAULT_SEED;

    @Option(longName = "crf-minibatch", usage = "Sets the minibatch size in the CRF trainer.")
    private int minibatchSize = 1;

    /**
     * Returns the configured CRF trainer.
     * @return The CRF trainer.
     */
    public CRFTrainer getSequenceTrainer() {
        return new CRFTrainer(sgoOptions.getOptimiser(), epochs, loggingInterval, minibatchSize, seed);
    }

}
