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

package org.tribuo.regression.sgd.linear;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.common.sgd.AbstractLinearSGDTrainer;
import org.tribuo.common.sgd.SGDObjective;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.math.la.DenseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.sgd.RegressionObjective;

import java.util.logging.Logger;

/**
 * A trainer for a linear regression model which uses SGD.
 * Independently trains each output dimension, unless they are tied together in the
 * optimiser.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public class LinearSGDTrainer extends AbstractLinearSGDTrainer<Regressor,DenseVector,LinearSGDModel> {
    private static final Logger logger = Logger.getLogger(LinearSGDTrainer.class.getName());

    @Config(mandatory = true,description="The regression objective to use.")
    private RegressionObjective objective;

    /**
     * Constructs an SGD trainer for a linear model.
     * @param objective The objective function to optimise.
     * @param optimiser The gradient optimiser to use.
     * @param epochs The number of epochs (complete passes through the training data).
     * @param loggingInterval Log the loss after this many iterations. If -1 don't log anything.
     * @param minibatchSize The size of any minibatches.
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public LinearSGDTrainer(RegressionObjective objective, StochasticGradientOptimiser optimiser, int epochs, int loggingInterval, int minibatchSize, long seed) {
        super(optimiser,epochs,loggingInterval,minibatchSize,seed);
        this.objective = objective;
    }

    /**
     * Constructs an SGD trainer for a linear model.
     * <p>
     * Sets the minibatch size to 1.
     * @param objective The objective function to optimise.
     * @param optimiser The gradient optimiser to use.
     * @param epochs The number of epochs (complete passes through the training data).
     * @param loggingInterval Log the loss after this many iterations. If -1 don't log anything.
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public LinearSGDTrainer(RegressionObjective objective, StochasticGradientOptimiser optimiser, int epochs, int loggingInterval, long seed) {
        this(objective,optimiser,epochs,loggingInterval,1,seed);
    }

    /**
     * Constructs an SGD trainer for a linear model.
     * <p>
     * Sets the minibatch size to 1 and the logging interval to 1000.
     * @param objective The objective function to optimise.
     * @param optimiser The gradient optimiser to use.
     * @param epochs The number of epochs (complete passes through the training data).
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public LinearSGDTrainer(RegressionObjective objective, StochasticGradientOptimiser optimiser, int epochs, long seed) {
        this(objective,optimiser,epochs,1000,1,seed);
    }

    /**
     * For olcut.
     */
    private LinearSGDTrainer() {
        super();
    }

    @Override
    protected DenseVector getTarget(ImmutableOutputInfo<Regressor> outputInfo, Regressor output) {
        double[] regressorsBuffer = new double[outputInfo.size()];
        for (Regressor.DimensionTuple r : output) {
            int id = outputInfo.getID(r);
            regressorsBuffer[id] = r.getValue();
        }
        return DenseVector.createDenseVector(regressorsBuffer);
    }

    @Override
    protected SGDObjective<DenseVector> getObjective() {
        return objective;
    }

    @Override
    protected LinearSGDModel createModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureMap, ImmutableOutputInfo<Regressor> outputInfo, LinearParameters parameters) {
        String[] dimensionNames = new String[outputInfo.size()];
        for (Regressor r : outputInfo.getDomain()) {
            int id = outputInfo.getID(r);
            dimensionNames[id] = r.getNames()[0];
        }
        return new LinearSGDModel(name,dimensionNames,provenance,featureMap,outputInfo,parameters);
    }

    @Override
    protected String getModelClassName() {
        return LinearSGDModel.class.getName();
    }

    @Override
    public String toString() {
        return "LinearSGDTrainer(objective="+objective.toString()+",optimiser="+optimiser.toString()+",epochs="+epochs+",minibatchSize="+minibatchSize+",seed="+seed+")";
    }
}
