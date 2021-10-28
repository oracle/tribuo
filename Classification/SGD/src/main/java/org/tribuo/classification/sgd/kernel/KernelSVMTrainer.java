/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.sgd.kernel;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.WeightedExamples;
import org.tribuo.classification.Label;
import org.tribuo.classification.sgd.Util;
import org.tribuo.math.kernel.Kernel;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * A trainer for a kernelised model using the Pegasos optimiser.
 * <p>
 * The Pegasos optimiser is extremely sensitive to the lambda parameter, and this
 * value must be tuned to get good performance.
 * <p>
 * See:
 * <pre>
 * Shalev-Shwartz S, Singer Y, Srebro N, Cotter A
 * "Pegasos: Primal Estimated Sub-Gradient Solver for SVM"
 * Mathematical Programming, 2011.
 * </pre>
 */
public class KernelSVMTrainer implements Trainer<Label>, WeightedExamples {
    private static final Logger logger = Logger.getLogger(KernelSVMTrainer.class.getName());

    @Config(mandatory = true,description="SVM kernel.")
    private Kernel kernel;

    @Config(mandatory = true,description="Step size.")
    private double lambda;

    @Config(description="Number of SGD epochs.")
    private int epochs = 5;

    @Config(description="Log values after this many updates.")
    private int loggingInterval = -1;

    @Config(mandatory = true,description="Seed for the RNG used to shuffle elements.")
    private long seed;

    @Config(description="Shuffle the data before each epoch. Only turn off for debugging.")
    private boolean shuffle = true;

    private SplittableRandom rng;

    private int trainInvocationCounter;

    /**
     * Constructs a trainer for a kernel SVM model.
     * @param kernel The kernel function to use as a similarity measure.
     * @param epochs The number of epochs (complete passes through the training data).
     * @param lambda l2 regulariser on the support vectors.
     * @param loggingInterval Log the loss after this many iterations. If -1 don't log anything.
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public KernelSVMTrainer(Kernel kernel, double lambda, int epochs, int loggingInterval, long seed) {
        this.kernel = kernel;
        this.lambda = lambda;
        this.epochs = epochs;
        this.loggingInterval = loggingInterval;
        this.seed = seed;
        postConfig();
    }

    /**
     * Constructs a trainer for a kernel SVM model.
     * Sets the logging interval to 1000.
     * @param kernel The kernel function to use as a similarity measure.
     * @param lambda l2 regulariser on the support vectors.
     * @param epochs The number of epochs (complete passes through the training data).
     * @param seed A seed for the random number generator, used to shuffle the examples before each epoch.
     */
    public KernelSVMTrainer(Kernel kernel, double lambda, int epochs, long seed) {
        this(kernel,lambda,epochs,1000,seed);
    }

    /**
     * For olcut.
     */
    private KernelSVMTrainer() { }

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
     * This method should only be used for debugging.
     * @param shuffle If true shuffle the examples, if false leave them in their current order.
     */
    public void setShuffle(boolean shuffle) {
        this.shuffle = shuffle;
    }

    @Override
    public KernelSVMModel train(Dataset<Label> examples, Map<String, Provenance> runProvenance) {
        return(train(examples, runProvenance, INCREMENT_INVOCATION_COUNT));
    }

    @Override
    public KernelSVMModel train(Dataset<Label> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // Creates a new RNG, adds one to the invocation count.
        TrainerProvenance trainerProvenance;
        SplittableRandom localRNG;
        synchronized(this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableOutputInfo<Label> labelIDMap = examples.getOutputIDInfo();
        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        SparseVector[] sgdFeatures = new SparseVector[examples.size()];
        int[] sgdLabels = new int[examples.size()];
        double[] weights = new double[examples.size()];
        int[] indices = new int[examples.size()];
        int n = 0;
        for (Example<Label> example : examples) {
            weights[n] = example.getWeight();
            sgdFeatures[n] = SparseVector.createSparseVector(example,featureIDMap,true);
            sgdLabels[n] = labelIDMap.getID(example.getOutput());
            indices[n] = n;
            n++;
        }
        logger.info(String.format("Training Kernel SVM with %d examples", n));
        logger.info(labelIDMap.toReadableString());

        double loss = 0.0;
        int iteration = 0;
        Map<Integer,SparseVector> supportVectors = new HashMap<>();
        double[][] alphas = new double[labelIDMap.size()][examples.size()];

        for (int i = 0; i < epochs; i++) {
            if (shuffle) {
                Util.shuffleInPlace(sgdFeatures, sgdLabels, weights, indices, localRNG);
            }
            for (int j = 0; j < sgdFeatures.length; j++) {
                SGDVector pred = predict(sgdFeatures[j],supportVectors,alphas);
                pred.add(sgdLabels[j],-1.0);
                int predIndex = pred.indexOfMax();

                if (sgdLabels[j] != predIndex) {
                    loss += (pred.get(sgdLabels[j]) - pred.get(predIndex)) * weights[j];
                    supportVectors.putIfAbsent(indices[j],sgdFeatures[j]);
                    alphas[sgdLabels[j]][indices[j]] += weights[j];
                }

                iteration++;
                if ((loggingInterval != -1) && (iteration % loggingInterval == 0)) {
                    logger.info("At iteration " + iteration + ", average loss = " + loss/loggingInterval + " with " + supportVectors.size() + " support vectors.");
                    loss = 0.0;
                }
            }
            logger.fine("Finished epoch " + i);
        }

        DenseMatrix alphaMatrix = new DenseMatrix(alphas.length,supportVectors.size());
        for (int i = 0; i < alphas.length; i++) {
            int rowCounter = 0;
            for (int j = 0; j < sgdFeatures.length; j++) {
                if (supportVectors.containsKey(j)) {
                    alphaMatrix.set(i, rowCounter, alphas[i][j]);
                    rowCounter++;
                }
            }
        }

        int counter = 0;
        SparseVector[] supportArray = new SparseVector[supportVectors.size()];
        for (int i = 0; i < sgdFeatures.length; i++) {
            SparseVector value = supportVectors.get(i);
            if (value != null) {
                supportArray[counter] = value;
                counter++;
            }
        }

        ModelProvenance provenance = new ModelProvenance(KernelSVMModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        //public KernelSVMModel(String name, String description, ImmutableInfoMap featureIDMap, ImmutableInfoMap outputIDInfo, Kernel kernel, SparseVector[] supportVectors, DenseMatrix weights)
        KernelSVMModel model = new KernelSVMModel("kernel-model",provenance,featureIDMap,labelIDMap,kernel,supportArray,alphaMatrix);
        return model;
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public synchronized void setInvocationCount(int invocationCount){
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        rng = new SplittableRandom(seed);

        for (trainInvocationCounter = 0; trainInvocationCounter < invocationCount; trainInvocationCounter++){
            SplittableRandom localRNG = rng.split();
        }

    }

    @Override
    public String toString() {
        return "KernelSVMTrainer(kernel="+kernel.toString()+",lambda="+lambda+",epochs="+epochs+",seed="+seed+")";
    }

    private SGDVector predict(SparseVector features, Map<Integer,SparseVector> sv, double[][] alphas) {
        double[] score = new double[alphas.length];

        for (Map.Entry<Integer, SparseVector> e : sv.entrySet()) {
            double distance = kernel.similarity(features,e.getValue());
            for (int i = 0; i < alphas.length; i++) {
                score[i] += alphas[i][e.getKey()] * distance;
            }
        }

        return DenseVector.createDenseVector(score);
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
