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

import org.tribuo.Trainer;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.math.optimisers.AdaGrad;

import java.util.logging.Logger;

/**
 * A logistic regression trainer that uses a reasonable objective, optimiser,
 * number of epochs and minibatch size. If you wish to modify any of these 
 * aspects, you can create your own LinearSGDTrainer.
 * <p>
 * This is strictly a convenience class for folks who are looking for
 * a simple logistic regression.
 */
public class LogisticRegressionTrainer extends LinearSGDTrainer {
    private static final Logger logger = Logger.getLogger(LogisticRegressionTrainer.class.getName());

    /**
     * Constructs a simple logistic regression, using {@link AdaGrad} with a learning rate of 1.0 as
     * the gradient optimizer, training for 5 epochs.
     *
     * It's equivalent to this:
     * {@code new LinearSGDTrainer(new LogMulticlass(), new AdaGrad(1.0, 0.1), 5, Trainer.DEFAULT_SEED); }
     */
    public LogisticRegressionTrainer() {
        super(new LogMulticlass(), new AdaGrad(1.0, 0.1), 5, Trainer.DEFAULT_SEED);
    }

}
