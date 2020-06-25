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
import java.util.Map;
import java.util.SplittableRandom;
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
 */
public class BaggingTrainer<T extends Output<T>> implements Trainer<T> {
    
    private static final Logger logger = Logger.getLogger(BaggingTrainer.class.getName());

    @Config(mandatory=true, description="The trainer to use for each ensemble member.")
    protected Trainer<T> innerTrainer;

    @Config(mandatory=true, description="The number of ensemble members to train.")
    protected int numMembers;

    @Config(mandatory=true, description="The seed for the RNG.")
    protected long seed;

    @Config(mandatory=true, description="The combination function to aggregate each ensemble member's outputs.")
    protected EnsembleCombiner<T> combiner;

    protected SplittableRandom rng;

    protected int trainInvocationCounter;

    /**
     * For the configuration system.
     */
    protected BaggingTrainer() { }

    public BaggingTrainer(Trainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers) {
        this(trainer, combiner, numMembers, Trainer.DEFAULT_SEED);
    }

    public BaggingTrainer(Trainer<T> trainer, EnsembleCombiner<T> combiner, int numMembers, long seed) {
        this.innerTrainer = trainer;
        this.combiner = combiner;
        this.numMembers = numMembers;
        this.seed = seed;
        postConfig();
    }

    @Override
    public synchronized void postConfig() {
        this.rng = new SplittableRandom(seed);
    }

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
    public Model<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance) {
        // Creates a new RNG, adds one to the invocation count.
        SplittableRandom localRNG;
        TrainerProvenance trainerProvenance;
        synchronized(this) {
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableFeatureMap featureIDs = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> labelIDs = examples.getOutputIDInfo();
        ArrayList<Model<T>> models = new ArrayList<>();
        for (int i = 0; i < numMembers; i++) {
            logger.info("Building model " + i);
            models.add(trainSingleModel(examples,featureIDs,labelIDs,localRNG,runProvenance));
        }
        EnsembleModelProvenance provenance = new EnsembleModelProvenance(WeightedEnsembleModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance, ListProvenance.createListProvenance(models));
        return new WeightedEnsembleModel<>(ensembleName(),provenance,featureIDs,labelIDs,models,combiner);
    }

    protected Model<T> trainSingleModel(Dataset<T> examples, ImmutableFeatureMap featureIDs, ImmutableOutputInfo<T> labelIDs, SplittableRandom localRNG, Map<String,Provenance> runProvenance) {
        DatasetView<T> bag = DatasetView.createBootstrapView(examples,examples.size(),localRNG.nextInt(),featureIDs,labelIDs);
        Model<T> newModel = innerTrainer.train(bag,runProvenance);
        return newModel;
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
