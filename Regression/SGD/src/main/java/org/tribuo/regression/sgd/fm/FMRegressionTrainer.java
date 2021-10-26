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

package org.tribuo.regression.sgd.fm;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.common.sgd.AbstractFMTrainer;
import org.tribuo.common.sgd.FMParameters;
import org.tribuo.common.sgd.SGDObjective;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.math.la.DenseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.ImmutableRegressionInfo;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.sgd.RegressionObjective;

import java.util.logging.Logger;

/**
 * A trainer for a regression factorization machine using SGD.
 * Independently trains each output dimension, unless they are tied together in the
 * optimiser.
 * <p>
 * See:
 * <pre>
 * Rendle, S.
 * Factorization machines.
 * 2010 IEEE International Conference on Data Mining
 * </pre>
 */
public class FMRegressionTrainer extends AbstractFMTrainer<Regressor, DenseVector, FMRegressionModel> {
    private static final Logger logger = Logger.getLogger(FMRegressionTrainer.class.getName());

    @Config(mandatory = true, description = "The regression objective to use.")
    private RegressionObjective objective;

    @Config(mandatory = true, description = "Standardise the output variables before fitting the model.")
    private boolean standardise;

    /**
     * Constructs an SGD trainer for a factorization machine.
     *
     * @param objective         The objective function to optimise.
     * @param optimiser         The gradient optimiser to use.
     * @param epochs            The number of epochs (complete passes through the training data).
     * @param loggingInterval   Log the loss after this many iterations. If -1 don't log anything.
     * @param minibatchSize     The size of any minibatches.
     * @param seed              A seed for the random number generator, used to shuffle the examples before each epoch.
     * @param factorizedDimSize Size of the factorized feature representation.
     * @param variance          The variance of the initializer.
     * @param standardise       Standardise the output regressors before fitting the model.
     */
    public FMRegressionTrainer(RegressionObjective objective, StochasticGradientOptimiser optimiser, int epochs,
                               int loggingInterval, int minibatchSize, long seed,
                               int factorizedDimSize, double variance, boolean standardise) {
        super(optimiser, epochs, loggingInterval, minibatchSize, seed, factorizedDimSize, variance);
        this.objective = objective;
        this.standardise = standardise;
    }

    /**
     * Constructs an SGD trainer for a factorization machine.
     * <p>
     * Sets the minibatch size to 1.
     *
     * @param objective         The objective function to optimise.
     * @param optimiser         The gradient optimiser to use.
     * @param epochs            The number of epochs (complete passes through the training data).
     * @param loggingInterval   Log the loss after this many iterations. If -1 don't log anything.
     * @param seed              A seed for the random number generator, used to shuffle the examples before each epoch.
     * @param factorizedDimSize Size of the factorized feature representation.
     * @param variance          The variance of the initializer.
     * @param standardise       Standardise the output regressors before fitting the model.
     */
    public FMRegressionTrainer(RegressionObjective objective, StochasticGradientOptimiser optimiser, int epochs,
                               int loggingInterval, long seed,
                               int factorizedDimSize, double variance, boolean standardise) {
        this(objective, optimiser, epochs, loggingInterval, 1, seed, factorizedDimSize, variance, standardise);
    }

    /**
     * Constructs an SGD trainer for a factorization machine.
     * <p>
     * Sets the minibatch size to 1 and the logging interval to 1000.
     *
     * @param objective         The objective function to optimise.
     * @param optimiser         The gradient optimiser to use.
     * @param epochs            The number of epochs (complete passes through the training data).
     * @param seed              A seed for the random number generator, used to shuffle the examples before each epoch.
     * @param factorizedDimSize Size of the factorized feature representation.
     * @param variance          The variance of the initializer.
     * @param standardise       Standardise the output regressors before fitting the model.
     */
    public FMRegressionTrainer(RegressionObjective objective, StochasticGradientOptimiser optimiser, int epochs,
                               long seed, int factorizedDimSize, double variance, boolean standardise) {
        this(objective, optimiser, epochs, 1000, 1, seed, factorizedDimSize, variance, standardise);
    }

    /**
     * For olcut.
     */
    private FMRegressionTrainer() {
        super();
    }

    @Override
    protected DenseVector getTarget(ImmutableOutputInfo<Regressor> outputInfo, Regressor output) {
        ImmutableRegressionInfo regressionInfo = (ImmutableRegressionInfo) outputInfo;
        double[] regressorsBuffer = new double[outputInfo.size()];
        for (Regressor.DimensionTuple r : output) {
            int id = outputInfo.getID(r);
            double curValue = r.getValue();
            if (standardise) {
                curValue = (curValue - regressionInfo.getMean(id)) / regressionInfo.getVariance(id);
            }
            regressorsBuffer[id] = curValue;
        }
        return DenseVector.createDenseVector(regressorsBuffer);
    }

    @Override
    protected SGDObjective<DenseVector> getObjective() {
        return objective;
    }

    @Override
    protected FMRegressionModel createModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureMap, ImmutableOutputInfo<Regressor> outputInfo, FMParameters parameters) {
        String[] dimensionNames = new String[outputInfo.size()];
        for (Regressor r : outputInfo.getDomain()) {
            int id = outputInfo.getID(r);
            dimensionNames[id] = r.getNames()[0];
        }
        return new FMRegressionModel(name, dimensionNames, provenance, featureMap, outputInfo, parameters, standardise);
    }

    @Override
    protected String getModelClassName() {
        return FMRegressionModel.class.getName();
    }

    @Override
    public String toString() {
        return "FMRegressionTrainer(objective=" + objective.toString() + ",optimiser=" + optimiser.toString() +
                ",epochs=" + epochs + ",minibatchSize=" + minibatchSize + ",seed=" + seed +
                ",factorizedDimSize=" + factorizedDimSize + ",variance=" + variance +
                ",standardise=" + standardise + ")";
    }
}
