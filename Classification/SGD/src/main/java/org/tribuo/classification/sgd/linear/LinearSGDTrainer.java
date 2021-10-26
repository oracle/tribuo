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

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.sgd.LabelObjective;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.common.sgd.AbstractLinearSGDTrainer;
import org.tribuo.common.sgd.SGDObjective;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.provenance.ModelProvenance;

import java.util.logging.Logger;

/**
 * A trainer for a linear classifier using SGD.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public class LinearSGDTrainer extends AbstractLinearSGDTrainer<Label,Integer,LinearSGDModel> {
    private static final Logger logger = Logger.getLogger(LinearSGDTrainer.class.getName());

    @Config(description = "The classification objective function to use.")
    private LabelObjective objective = new LogMulticlass();

    /**
     * Constructs an SGD trainer for a linear model.
     *
     * @param objective       The objective function to optimise.
     * @param optimiser       The gradient optimiser to use.
     * @param epochs          The number of epochs (complete passes through the training data).
     * @param loggingInterval Log the loss after this many iterations. If -1 don't log anything.
     * @param minibatchSize   The size of any minibatches.
     * @param seed            A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public LinearSGDTrainer(LabelObjective objective, StochasticGradientOptimiser optimiser, int epochs, int loggingInterval, int minibatchSize, long seed) {
        super(optimiser, epochs, loggingInterval, minibatchSize, seed);
        this.objective = objective;
    }

    /**
     * Constructs an SGD trainer for a linear model.
     * <p>
     * Sets the minibatch size to 1.
     *
     * @param objective       The objective function to optimise.
     * @param optimiser       The gradient optimiser to use.
     * @param epochs          The number of epochs (complete passes through the training data).
     * @param loggingInterval Log the loss after this many iterations. If -1 don't log anything.
     * @param seed            A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public LinearSGDTrainer(LabelObjective objective, StochasticGradientOptimiser optimiser, int epochs, int loggingInterval, long seed) {
        this(objective, optimiser, epochs, loggingInterval, 1, seed);
    }

    /**
     * Constructs an SGD trainer for a linear model.
     * <p>
     * Sets the minibatch size to 1 and the logging interval to 1000.
     *
     * @param objective The objective function to optimise.
     * @param optimiser The gradient optimiser to use.
     * @param epochs    The number of epochs (complete passes through the training data).
     * @param seed      A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public LinearSGDTrainer(LabelObjective objective, StochasticGradientOptimiser optimiser, int epochs, long seed) {
        this(objective, optimiser, epochs, 1000, 1, seed);
    }

    /**
     * For olcut.
     */
    private LinearSGDTrainer() {
        super();
    }

    @Override
    protected Integer getTarget(ImmutableOutputInfo<Label> outputInfo, Label output) {
        return outputInfo.getID(output);
    }

    @Override
    protected SGDObjective<Integer> getObjective() {
        return objective;
    }

    @Override
    protected LinearSGDModel createModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureMap, ImmutableOutputInfo<Label> outputInfo, LinearParameters parameters) {
        return new LinearSGDModel(name, provenance, featureMap, outputInfo, parameters, objective.getNormalizer(), objective.isProbabilistic());
    }

    @Override
    protected String getModelClassName() {
        return LinearSGDModel.class.getName();
    }

    @Override
    public String toString() {
        return "LinearSGDTrainer(objective=" + objective.toString() + ",optimiser=" + optimiser.toString() + ",epochs=" + epochs + ",minibatchSize=" + minibatchSize + ",seed=" + seed + ")";
    }
}
