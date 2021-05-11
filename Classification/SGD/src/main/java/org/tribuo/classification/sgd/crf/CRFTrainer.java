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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.WeightedExamples;
import org.tribuo.classification.Label;
import org.tribuo.classification.sgd.Util;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.sequence.SequenceDataset;
import org.tribuo.sequence.SequenceExample;
import org.tribuo.sequence.SequenceTrainer;

import java.time.OffsetDateTime;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * A trainer for CRFs using SGD. Modelled after FACTORIE's trainer for CRFs.
 * <p>
 * See:
 * <pre>
 * Lafferty J, McCallum A, Pereira FC.
 * "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"
 * Proceedings of the 18th International Conference on Machine Learning 2001 (ICML 2001).
 * </pre>
 */
public class CRFTrainer implements SequenceTrainer<Label>, WeightedExamples {
    private static final Logger logger = Logger.getLogger(CRFTrainer.class.getName());

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
     * Creates a CRFTrainer which uses SGD to learn the parameters.
     * @param optimiser The gradient optimiser to use.
     * @param epochs The number of SGD epochs (complete passes through the training data).
     * @param loggingInterval Log the loss after this many iterations. If -1 don't log anything.
     * @param minibatchSize The size of the minibatches used to aggregate gradients.
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public CRFTrainer(StochasticGradientOptimiser optimiser, int epochs, int loggingInterval, int minibatchSize, long seed) {
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
     * @param epochs The number of SGD epochs (complete passes through the training data).
     * @param loggingInterval Log the loss after this many iterations. If -1 don't log anything.
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public CRFTrainer(StochasticGradientOptimiser optimiser, int epochs, int loggingInterval, long seed) {
        this(optimiser,epochs,loggingInterval,1,seed);
    }

    /**
     * Sets the minibatch size to 1 and the logging interval to 100.
     * @param optimiser The gradient optimiser to use.
     * @param epochs The number of SGD epochs (complete passes through the training data).
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public CRFTrainer(StochasticGradientOptimiser optimiser, int epochs, long seed) {
        this(optimiser,epochs,100,1,seed);
    }

    /**
     * For olcut.
     */
    private CRFTrainer() { }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
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
    public CRFModel train(SequenceDataset<Label> sequenceExamples, Map<String, Provenance> runProvenance) {
        if (sequenceExamples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // Creates a new RNG, adds one to the invocation count, generates a local optimiser.
        SplittableRandom localRNG;
        TrainerProvenance trainerProvenance;
        StochasticGradientOptimiser localOptimiser;
        synchronized(this) {
            localRNG = rng.split();
            localOptimiser = optimiser.copy();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableOutputInfo<Label> labelIDMap = sequenceExamples.getOutputIDInfo();
        ImmutableFeatureMap featureIDMap = sequenceExamples.getFeatureIDMap();
        SGDVector[][] sgdFeatures = new SGDVector[sequenceExamples.size()][];
        int[][] sgdLabels = new int[sequenceExamples.size()][];
        double[] weights = new double[sequenceExamples.size()];
        int n = 0;
        for (SequenceExample<Label> example : sequenceExamples) {
            weights[n] = example.getWeight();
            Pair<int[],SGDVector[]> pair = CRFModel.convertToVector(example,featureIDMap,labelIDMap);
            sgdFeatures[n] = pair.getB();
            sgdLabels[n] = pair.getA();
            n++;
        }
        logger.info(String.format("Training SGD CRF with %d examples", n));

        CRFParameters crfParameters = new CRFParameters(featureIDMap.size(),labelIDMap.size());

        localOptimiser.initialise(crfParameters);
        double loss = 0.0;
        int iteration = 0;

        for (int i = 0; i < epochs; i++) {
            if (shuffle) {
                Util.shuffleInPlace(sgdFeatures, sgdLabels, weights, localRNG);
            }
            if (minibatchSize == 1) {
                /*
                 * Special case a minibatch of size 1. Directly updates the parameters after each
                 * example rather than aggregating.
                 */
                for (int j = 0; j < sgdFeatures.length; j++) {
                    Pair<Double,Tensor[]> output = crfParameters.valueAndGradient(sgdFeatures[j],sgdLabels[j]);
                    loss += output.getA()*weights[j];

                    //Update the gradient with the current learning rates
                    Tensor[] updates = localOptimiser.step(output.getB(),weights[j]);

                    //Apply the update to the current parameters.
                    crfParameters.update(updates);

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
                    //Aggregate the gradient updates for each example in the minibatch
                    for (int k = j; k < j+minibatchSize && k < sgdFeatures.length; k++) {
                        Pair<Double,Tensor[]> output = crfParameters.valueAndGradient(sgdFeatures[j],sgdLabels[j]);
                        loss += output.getA()*weights[k];
                        tempWeight += weights[k];

                        gradients[k-j] = output.getB();
                        curSize++;
                    }
                    //Merge the values into a single gradient update
                    Tensor[] updates = crfParameters.merge(gradients,curSize);
                    for (Tensor update : updates) {
                        update.scaleInPlace(minibatchSize);
                    }
                    tempWeight /= minibatchSize;
                    //Update the gradient with the current learning rates
                    updates = localOptimiser.step(updates,tempWeight);
                    //Apply the gradient.
                    crfParameters.update(updates);

                    iteration++;
                    if ((loggingInterval != -1) && (iteration % loggingInterval == 0)) {
                        logger.info("At iteration " + iteration + ", average loss = " + loss/loggingInterval);
                        loss = 0.0;
                    }
                }
            }
        }
        localOptimiser.finalise();
        //public CRFModel(String name, String description, ImmutableInfoMap featureIDMap, ImmutableInfoMap outputIDInfo, CRFParameters parameters) {
        ModelProvenance provenance = new ModelProvenance(CRFModel.class.getName(),OffsetDateTime.now(),sequenceExamples.getProvenance(),trainerProvenance,runProvenance);
        CRFModel model = new CRFModel("crf-sgd-model",provenance,featureIDMap,labelIDMap,crfParameters);
        localOptimiser.reset();
        return model;
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public String toString() {
        return "CRFTrainer(optimiser="+optimiser.toString()+",epochs="+epochs+",minibatchSize="+minibatchSize+",seed="+seed+")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
