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
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.WeightedExamples;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.sgd.RegressionObjective;
import org.tribuo.regression.sgd.Util;

import java.time.OffsetDateTime;
import java.util.Arrays;
import java.util.Map;
import java.util.SplittableRandom;
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
public class LinearSGDTrainer implements Trainer<Regressor>, WeightedExamples {
    private static final Logger logger = Logger.getLogger(LinearSGDTrainer.class.getName());

    @Config(mandatory = true,description="The regression objective to use.")
    private RegressionObjective objective;

    @Config(mandatory = true,description="The gradient optimiser to use.")
    private StochasticGradientOptimiser optimiser;

    @Config(description="The number of gradient descent epochs.")
    private int epochs = 5;

    @Config(description="Log values after this many updates.")
    private int loggingInterval = -1;

    @Config(description="Minibatch size in SGD.")
    private int minibatchSize = 1;

    @Config(mandatory = true,description="Seed for the RNG used to shuffle elements.")
    private long seed;

    @Config(description="Shuffle the data before each epoch. Only turn off for debugging.")
    private boolean shuffle = true;

    private SplittableRandom rng;

    private int trainInvocationCounter;

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
        this.objective = objective;
        this.optimiser = optimiser;
        this.epochs = epochs;
        this.loggingInterval = loggingInterval;
        this.minibatchSize = minibatchSize;
        this.seed = seed;
        postConfig();
    }

    /**
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
    private LinearSGDTrainer() { }

    @Override
    public synchronized void postConfig() {
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Turn on or off shuffling of examples.
     * <p>
     * This isn't exposed in the constructor as it defaults to on.
     * This method should be used for debugging.
     * @param shuffle If true shuffle the examples, if false leave them in their current order.
     */
    public void setShuffle(boolean shuffle) {
        this.shuffle = shuffle;
    }

    @Override
    public LinearSGDModel train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // Creates a new RNG, adds one to the invocation count, generates a local optimiser.
        TrainerProvenance trainerProvenance;
        SplittableRandom localRNG;
        StochasticGradientOptimiser localOptimiser;
        synchronized(this) {
            localRNG = rng.split();
            localOptimiser = optimiser.copy();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableOutputInfo<Regressor> outputInfo = examples.getOutputIDInfo();
        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        int numOutputs = outputInfo.size();
        SparseVector[] sgdFeatures = new SparseVector[examples.size()];
        DenseVector[] sgdLabels = new DenseVector[examples.size()];
        double[] weights = new double[examples.size()];
        int n = 0;
        double[] regressorsBuffer = new double[numOutputs];
        for (Example<Regressor> example : examples) {
            weights[n] = example.getWeight();
            sgdFeatures[n] = SparseVector.createSparseVector(example,featureIDMap,true);
            Arrays.fill(regressorsBuffer,0.0);
            for (Regressor.DimensionTuple r : example.getOutput()) {
                int id = outputInfo.getID(r);
                regressorsBuffer[id] = r.getValue();
            }
            sgdLabels[n] = DenseVector.createDenseVector(regressorsBuffer);
            n++;
        }
        String[] dimensionNames = new String[numOutputs];
        for (Regressor r : outputInfo.getDomain()) {
            int id = outputInfo.getID(r);
            dimensionNames[id] = r.getNames()[0];
        }
        logger.info(String.format("Training SGD regressor with %d examples", n));
        logger.info("Output variable " + outputInfo.toReadableString());

        // featureIDMap.size()+1 adds the bias feature.
        LinearParameters linearParameters = new LinearParameters(featureIDMap.size()+1,dimensionNames.length);

        localOptimiser.initialise(linearParameters);
        double loss = 0.0;
        int iteration = 0;

        for (int i = 0; i < epochs; i++) {
            if (shuffle) {
                Util.shuffleInPlace(sgdFeatures, sgdLabels, weights, localRNG);
            }
            if (minibatchSize == 1) {
                for (int j = 0; j < sgdFeatures.length; j++) {
                    SGDVector pred = linearParameters.predict(sgdFeatures[j]);
                    Pair<Double,SGDVector> output = objective.loss(sgdLabels[j],pred);
                    loss += output.getA()*weights[j];

                    Tensor[] updates = localOptimiser.step(linearParameters.gradients(output,sgdFeatures[j]),weights[j]);
                    linearParameters.update(updates);

                    iteration++;
                    if ((iteration % loggingInterval == 0) && (loggingInterval != -1)) {
                        logger.info("At iteration " + iteration + ", average loss = " + loss/loggingInterval);
                        loss = 0.0;
                    }
                }
            } else {
                Tensor[][] gradients = new Tensor[minibatchSize][];
                for (int j = 0; j < sgdFeatures.length; j += minibatchSize) {
                    double tempWeight = 0.0;
                    int curSize = 0;
                    for (int k = j; k < j+minibatchSize && k < sgdFeatures.length; k++) {
                        SGDVector pred = linearParameters.predict(sgdFeatures[k]);
                        Pair<Double,SGDVector> output = objective.loss(sgdLabels[k],pred);
                        loss += output.getA()*weights[k];
                        tempWeight += weights[k];

                        gradients[k-j] = linearParameters.gradients(output,sgdFeatures[k]);
                        curSize++;
                    }
                    Tensor[] updates = linearParameters.merge(gradients,curSize);
                    for (int k = 0; k < updates.length; k++) {
                        updates[k].scaleInPlace(minibatchSize);
                    }
                    tempWeight /= minibatchSize;
                    updates = localOptimiser.step(updates,tempWeight);
                    linearParameters.update(updates);

                    iteration++;
                    if ((loggingInterval != -1) && (iteration % loggingInterval == 0)) {
                        logger.info("At iteration " + iteration + ", average loss = " + loss/loggingInterval);
                        loss = 0.0;
                    }
                }
            }
        }
        localOptimiser.finalise();
        ModelProvenance provenance = new ModelProvenance(LinearSGDModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        LinearSGDModel model = new LinearSGDModel("linear-sgd-model",dimensionNames,provenance,featureIDMap,outputInfo,linearParameters);
        return model;
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public String toString() {
        return "LinearSGDTrainer(objective="+objective.toString()+",optimiser="+optimiser.toString()+",epochs="+epochs+",minibatchSize="+minibatchSize+",seed="+seed+")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
