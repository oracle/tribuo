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

package org.tribuo.ensemble;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.dataset.DatasetView;
import org.tribuo.provenance.EnsembleModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.logging.Logger;

/**
 * A Trainer that wraps another trainer and produces a bagged ensemble.
 * <p>
 * A bagged ensemble is a set of models each of which was trained on a bootstrap sample of the
 * original dataset, combined with an unweighted majority vote.
 * <p>
 * See:
 * <pre>
 * J. Friedman, T. Hastie, &amp; R. Tibshirani.
 * "The Elements of Statistical Learning"
 * Springer 2001. <a href="http://web.stanford.edu/~hastie/ElemStatLearn/">PDF</a>
 * </pre>
 * @param <T> The prediction type.
 */
public class BaggingTrainer<T extends Output<T>> implements Trainer<T> {

    private static final Logger logger = Logger.getLogger(BaggingTrainer.class.getName());

    /**
     * Constant for numThreads indicating to use all available processors
     * via Runtime.getRuntime().availableProcessors().
     */
    public static final int USE_ALL_AVAILABLE_PROCESSORS = -1;

    // Thread factory for ensemble training, uses low priority to avoid starving other operations
    private static final ThreadFactory LOW_PRIORITY_THREAD_FACTORY = r -> {
        Thread t = new Thread(r);
        t.setPriority(Thread.MIN_PRIORITY);
        return t;
    };

    @Config(mandatory=true, description="The trainer to use for each ensemble member.")
    protected Trainer<T> innerTrainer;

    @Config(mandatory=true, description="The number of ensemble members to train.")
    protected int numMembers;

    @Config(mandatory=true, description="The seed for the RNG.")
    protected long seed;

    @Config(mandatory=true, description="The combination function to aggregate each ensemble member's outputs.")
    protected EnsembleCombiner<T> combiner;

    @Config(description="The number of threads to use for training. Defaults to 1 (single-threaded). Set to BaggingTrainer.USE_ALL_AVAILABLE_PROCESSORS to use all available processors.")
    protected int numThreads = 1;

    protected SplittableRandom rng;

    protected int trainInvocationCounter;

    /**
     * For the configuration system.
     */
    protected BaggingTrainer() { }

    /**
     * Constructs a bagging trainer with the supplied parameters using {@link Trainer#DEFAULT_SEED} as the RNG seed.
     * @param trainer The ensemble member trainer.
     * @param combiner The combination function.
     * @param numMembers The number of ensemble members to train.
     */
    public BaggingTrainer(Trainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers) {
        this(trainer, combiner, numMembers, Trainer.DEFAULT_SEED);
    }

    /**
     * Constructs a bagging trainer with the supplied parameters.
     * @param trainer The ensemble member trainer.
     * @param combiner The combination function.
     * @param numMembers The number of ensemble members to train.
     * @param seed The RNG seed used to bootstrap the datasets.
     */
    public BaggingTrainer(Trainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers, long seed) {
        this.innerTrainer = trainer;
        this.combiner = combiner;
        this.numMembers = numMembers;
        this.seed = seed;
        this.numThreads = 1;
        postConfig();
    }

    /**
     * Constructs a bagging trainer with the supplied parameters and thread count.
     * @param trainer The ensemble member trainer.
     * @param combiner The combination function.
     * @param numMembers The number of ensemble members to train.
     * @param seed The RNG seed used to bootstrap the datasets.
     * @param numThreads The number of threads to use for parallel training. Set to USE_ALL_AVAILABLE_PROCESSORS to use all available processors, 1 for single-threaded.
     */
    public BaggingTrainer(Trainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers, long seed, int numThreads) {
        this.innerTrainer = trainer;
        this.combiner = combiner;
        this.numMembers = numMembers;
        this.seed = seed;
        this.numThreads = numThreads;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public synchronized void postConfig() {
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Default name of the generated ensemble.
     * @return The default ensemble name.
     */
    protected String ensembleName() {
        return "bagging-ensemble";
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("BaggingTrainer(");
        buffer.append("innerTrainer=");
        buffer.append(innerTrainer.toString());
        buffer.append(",combiner=");
        buffer.append(combiner.toString());
        buffer.append(",numMembers=");
        buffer.append(numMembers);
        buffer.append(",seed=");
        buffer.append(seed);
        buffer.append(")");

        return buffer.toString();
    }

    @Override
    public EnsembleModel<T> train(Dataset<T> examples) {
        return train(examples, Collections.emptyMap());
    }

    @Override
    public EnsembleModel<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance) {
        return train(examples, runProvenance, INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public EnsembleModel<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        // Creates a new RNG, adds one to the invocation count.
        SplittableRandom localRNG;
        TrainerProvenance trainerProvenance;
        synchronized(this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT){
                setInvocationCount(invocationCount);
            }
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableFeatureMap featureIDs = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> labelIDs = examples.getOutputIDInfo();

        // Determine number of threads
        int threads = (numThreads == USE_ALL_AVAILABLE_PROCESSORS) ? Runtime.getRuntime().availableProcessors() : numThreads;

        // Pre-generate all random seeds - this maintains reproducibility
        int[] seeds = new int[numMembers];
        for (int i = 0; i < numMembers; i++) {
            seeds[i] = localRNG.nextInt();
        }

        // Train models either single-threaded or multi-threaded
        ArrayList<Model<T>> models;
        if (threads == 1 || numMembers == 1) {
            models = trainModelsSequentially(examples, featureIDs, labelIDs, seeds, runProvenance);
        } else {
            models = trainModelsInParallel(examples, featureIDs, labelIDs, seeds, runProvenance, threads);
        }

        EnsembleModelProvenance provenance = new EnsembleModelProvenance(WeightedEnsembleModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance, ListProvenance.createListProvenance(models));
        return createEnsemble(provenance, featureIDs, labelIDs, models);
    }

    /**
     * Factory method to create the ensemble model.
     * Subclasses can override this to return optimized ensemble implementations.
     *
     * @param provenance The ensemble provenance.
     * @param featureIDs The feature domain.
     * @param labelIDs The output domain.
     * @param models The list of trained models.
     * @return An ensemble model.
     */
    protected EnsembleModel<T> createEnsemble(EnsembleModelProvenance provenance, ImmutableFeatureMap featureIDs, ImmutableOutputInfo<T> labelIDs, ArrayList<Model<T>> models) {
        return new WeightedEnsembleModel<>(ensembleName(), provenance, featureIDs, labelIDs, models, combiner);
    }

    /**
     * Trains models sequentially in a single thread.
     * @param examples The training dataset.
     * @param featureIDs The feature domain.
     * @param labelIDs The output domain.
     * @param seeds The random seeds for each model.
     * @param runProvenance Provenance for this instance.
     * @return The list of trained models.
     */
    private ArrayList<Model<T>> trainModelsSequentially(Dataset<T> examples, ImmutableFeatureMap featureIDs, ImmutableOutputInfo<T> labelIDs, int[] seeds, Map<String,Provenance> runProvenance) {
        ArrayList<Model<T>> models = new ArrayList<>();
        int initialInvocation = innerTrainer.getInvocationCount();
        for (int i = 0; i < numMembers; i++) {
            logger.info("Building model " + i);
            models.add(trainSingleModel(examples, featureIDs, labelIDs, seeds[i], runProvenance, initialInvocation + i));
        }
        return models;
    }

    /**
     * Trains models in parallel using multiple threads.
     * Each model is submitted as an individual task to the executor,
     * allowing for optimal load balancing across threads.
     * @param examples The training dataset.
     * @param featureIDs The feature domain.
     * @param labelIDs The output domain.
     * @param seeds The random seeds for each model.
     * @param runProvenance Provenance for this instance.
     * @param threads The number of threads to use.
     * @return The list of trained models.
     */
    private ArrayList<Model<T>> trainModelsInParallel(Dataset<T> examples, ImmutableFeatureMap featureIDs, ImmutableOutputInfo<T> labelIDs, int[] seeds, Map<String,Provenance> runProvenance, int threads) {
        // Use low priority threads so ensemble training doesn't prevent other operations from completing.
        // Model training is typically a background batch operation and shouldn't starve interactive requests.
        ExecutorService executor = Executors.newFixedThreadPool(threads, LOW_PRIORITY_THREAD_FACTORY);

        List<Future<Model<T>>> futures = new ArrayList<>(numMembers);
        int initialInvocation = innerTrainer.getInvocationCount();

        try {
            // Submit each model training as an individual task
            for (int i = 0; i < numMembers; i++) {
                final int idx = i;
                Future<Model<T>> future = executor.submit(() -> {
                    logger.info("Building model " + idx + " on thread " + Thread.currentThread().getName());
                    return trainSingleModel(examples, featureIDs, labelIDs, seeds[idx], runProvenance, initialInvocation + idx);
                });
                futures.add(future);
            }

            // Collect results in order
            ArrayList<Model<T>> models = new ArrayList<>(numMembers);
            for (Future<Model<T>> future : futures) {
                try {
                    models.add(future.get());
                } catch (InterruptedException | ExecutionException e) {
                    executor.shutdownNow();
                    throw new RuntimeException("Training failed", e);
                }
            }
            return models;
        } finally {
            executor.shutdown();
        }
    }

    /**
     * Trains a single model.
     * @param examples The training dataset.
     * @param featureIDs The feature domain.
     * @param labelIDs The output domain.
     * @param randInt A random int from an rng instance
     * @param runProvenance Provenance for this instance.
     * @param invocationCount The invocation count for the inner trainer.
     * @return The trained ensemble member.
     */
    protected Model<T> trainSingleModel(Dataset<T> examples, ImmutableFeatureMap featureIDs, ImmutableOutputInfo<T> labelIDs, int randInt, Map<String,Provenance> runProvenance, int invocationCount) {
        DatasetView<T> bag = DatasetView.createBootstrapView(examples,examples.size(),randInt,featureIDs,labelIDs);
        Model<T> newModel = innerTrainer.train(bag,runProvenance, invocationCount);
        return newModel;
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
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
