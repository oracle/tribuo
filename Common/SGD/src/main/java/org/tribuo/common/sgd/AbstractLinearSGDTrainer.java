/*
 * Copyright (c) 2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.sgd;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.WeightedExamples;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.Collections;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * A trainer for a linear model which uses SGD.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public abstract class AbstractLinearSGDTrainer<T extends Output<T>,U> implements Trainer<T>, WeightedExamples {
    private static final Logger logger = Logger.getLogger(AbstractLinearSGDTrainer.class.getName());

    @Config(description="The gradient optimiser to use.")
    protected StochasticGradientOptimiser optimiser = new AdaGrad(1.0,0.1);

    @Config(description="The number of gradient descent epochs.")
    protected int epochs = 5;

    @Config(description="Log values after this many updates.")
    protected int loggingInterval = -1;

    @Config(description="Minibatch size in SGD.")
    protected int minibatchSize = 1;

    @Config(description="Seed for the RNG used to shuffle elements.")
    protected long seed = Trainer.DEFAULT_SEED;

    @Config(description="Shuffle the data before each epoch. Only turn off for debugging.")
    protected boolean shuffle = true;

    protected SplittableRandom rng;

    private int trainInvocationCounter;

    /**
     * Constructs an SGD trainer for a linear model.
     * @param optimiser The gradient optimiser to use.
     * @param epochs The number of epochs (complete passes through the training data).
     * @param loggingInterval Log the loss after this many iterations. If -1 don't log anything.
     * @param minibatchSize The size of any minibatches.
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    protected AbstractLinearSGDTrainer(StochasticGradientOptimiser optimiser, int epochs, int loggingInterval, int minibatchSize, long seed) {
        this.optimiser = optimiser;
        this.epochs = epochs;
        this.loggingInterval = loggingInterval;
        this.minibatchSize = minibatchSize;
        this.seed = seed;
        postConfig();
    }

    /**
     * Sets the minibatch size to 1.
     * @param optimiser The gradient optimiser to use.
     * @param epochs The number of epochs (complete passes through the training data).
     * @param loggingInterval Log the loss after this many iterations. If -1 don't log anything.
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    protected AbstractLinearSGDTrainer(StochasticGradientOptimiser optimiser, int epochs, int loggingInterval, long seed) {
        this(optimiser,epochs,loggingInterval,1,seed);
    }

    /**
     * Sets the minibatch size to 1 and the logging interval to 1000.
     * @param optimiser The gradient optimiser to use.
     * @param epochs The number of epochs (complete passes through the training data).
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    protected AbstractLinearSGDTrainer(StochasticGradientOptimiser optimiser, int epochs, long seed) {
        this(optimiser,epochs,1000,1,seed);
    }

    /**
     * For olcut.
     */
    protected AbstractLinearSGDTrainer() { }

    @Override
    public synchronized void postConfig() {
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Turn on or off shuffling of examples.
     * <p>
     * This isn't exposed in the constructor as it defaults to on.
     * This method should only be used for debugging.
     * @param shuffle If true shuffle the examples, if false leave them in their current order.
     */
    public void setShuffle(boolean shuffle) {
        this.shuffle = shuffle;
    }

    @Override
    public AbstractLinearSGDModel<T> train(Dataset<T> examples) {
        return train(examples, Collections.emptyMap());
    }

    @Override
    public AbstractLinearSGDModel<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance) {
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

        SGDObjective<U> objective = getObjective();
        ImmutableOutputInfo<T> outputIDInfo = examples.getOutputIDInfo();
        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        SparseVector[] sgdFeatures = new SparseVector[examples.size()];
        @SuppressWarnings("unchecked")
        U[] sgdTargets = (U[]) new Object[examples.size()];
        double[] weights = new double[examples.size()];
        int n = 0;
        for (Example<T> example : examples) {
            weights[n] = example.getWeight();
            sgdFeatures[n] = SparseVector.createSparseVector(example,featureIDMap,true);
            sgdTargets[n] = getTarget(outputIDInfo,example.getOutput());
            n++;
        }
        logger.info(String.format("Training linear SGD model with %d examples", n));
        logger.info("Outputs - " + outputIDInfo.toReadableString());

        // featureIDMap.size()+1 adds the bias feature.
        LinearParameters linearParameters = new LinearParameters(featureIDMap.size()+1,outputIDInfo.size());

        localOptimiser.initialise(linearParameters);
        double loss = 0.0;
        int iteration = 0;

        for (int i = 0; i < epochs; i++) {
            if (shuffle) {
                shuffleInPlace(sgdFeatures, sgdTargets, weights, localRNG);
            }
            if (minibatchSize == 1) {
                for (int j = 0; j < sgdFeatures.length; j++) {
                    SGDVector pred = linearParameters.predict(sgdFeatures[j]);
                    Pair<Double,SGDVector> output = objective.lossAndGradient(sgdTargets[j],pred);
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
                        Pair<Double,SGDVector> output = objective.lossAndGradient(sgdTargets[k],pred);
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
        ModelProvenance provenance = new ModelProvenance(getModelClassName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        AbstractLinearSGDModel<T> model = createModel("linear-sgd-model",provenance,featureIDMap,outputIDInfo,linearParameters);
        localOptimiser.reset();
        return model;
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    /**
     * Extracts the appropriate training time representation from the supplied output.
     * @param outputInfo The output info to use.
     * @param output The output to extract.
     * @return The training time representation of the output.
     */
    protected abstract U getTarget(ImmutableOutputInfo<T> outputInfo, T output);

    /**
     * Returns the objective used by this trainer.
     * @return The SGDObjective used by this trainer.
     */
    protected abstract SGDObjective<U> getObjective();

    /**
     * Creates the appropriate model subclass for this subclass of AbstractLinearSGDTrainer.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureMap The feature map.
     * @param outputInfo The output info.
     * @param parameters The model parameters.
     * @return A new instance of the appropriate subclass of {@link Model}.
     */
    protected abstract AbstractLinearSGDModel<T> createModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureMap, ImmutableOutputInfo<T> outputInfo, LinearParameters parameters);

    /**
     * Returns the class name of the model that's produced by this trainer.
     * @return The model class name;
     */
    protected abstract String getModelClassName();

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

    /**
     * In place shuffle of the features, outputs and weights.
     * @param features Feature array.
     * @param labels Output array.
     * @param weights Weight array.
     * @param rng Random number generator.
     * @param <T> The output type.
     */
    public static <T> void shuffleInPlace(SGDVector[] features, T[] labels, double[] weights, SplittableRandom rng) {
        int size = features.length;
        // Shuffle array
        for (int i = size; i > 1; i--) {
            int j = rng.nextInt(i);
            //swap features
            SGDVector tmpFeature = features[i-1];
            features[i-1] = features[j];
            features[j] = tmpFeature;
            //swap labels
            T tmpLabel = labels[i-1];
            labels[i-1] = labels[j];
            labels[j] = tmpLabel;
            //swap weights
            double tmpWeight = weights[i-1];
            weights[i-1] = weights[j];
            weights[j] = tmpWeight;
        }
    }
}
