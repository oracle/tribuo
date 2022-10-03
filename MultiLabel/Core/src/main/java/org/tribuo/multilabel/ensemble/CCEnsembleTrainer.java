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

package org.tribuo.multilabel.ensemble;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.ensemble.WeightedEnsembleModel;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.baseline.ClassifierChainTrainer;
import org.tribuo.provenance.EnsembleModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * A trainer for an ensemble of randomly ordered Classifier Chains.
 * <p>
 * This ensemble is useful if there is no a-priori knowledge of the
 * label dependence structure, as it averages over many possible structures.
 * In addition, ensembling is frequently a powerful technique for improving
 * general classification performance.
 * <p>
 * {@link ClassifierChainTrainer} for more details on classifier chains.
 * <p>
 * See:
 * <pre>
 * Read, J., Pfahringer, B., Holmes, G., &amp; Frank, E.
 * "Classifier Chains for Multi-Label Classification"
 * Machine Learning, pages 333-359, 2011.
 * </pre>
 */
public final class CCEnsembleTrainer implements Trainer<MultiLabel> {
    private static final Logger logger = Logger.getLogger(CCEnsembleTrainer.class.getName());

    @Config(mandatory = true, description = "The trainer to use.")
    private Trainer<Label> innerTrainer;

    @Config(mandatory = true, description = "Number of classifier chains to build.")
    private int numMembers;

    @Config(mandatory = true, description = "RNG seed for random label orders.")
    private long seed;

    private int trainInvocationCounter = 0;
    private SplittableRandom rng;

    /**
     * For OLCUT.
     */
    private CCEnsembleTrainer() {}

    /**
     * Constructs a classifier chain ensemble trainer.
     * @param innerTrainer The trainer to use to train each chain model.
     * @param numMembers The number of chains to train.
     * @param seed The RNG seed.
     */
    public CCEnsembleTrainer(Trainer<Label> innerTrainer, int numMembers, long seed) {
        if (numMembers < 1) {
            throw new IllegalArgumentException("Must have a positive number of ensemble members, found " + numMembers);
        }
        this.innerTrainer = innerTrainer;
        this.numMembers = numMembers;
        this.seed = seed;
        postConfig();
    }

    @Override
    public void postConfig() throws PropertyException {
        if (numMembers < 1) {
            throw new PropertyException("","numMembers","Must have a positive number of ensemble members, found " + numMembers);
        }
        this.rng = new SplittableRandom(this.seed);
    }

    @Override
    public WeightedEnsembleModel<MultiLabel> train(Dataset<MultiLabel> examples) {
        return train(examples, Collections.emptyMap());
    }

    @Override
    public WeightedEnsembleModel<MultiLabel> train(Dataset<MultiLabel> examples, Map<String, Provenance> runProvenance) {
        return train(examples,runProvenance,INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public WeightedEnsembleModel<MultiLabel> train(Dataset<MultiLabel> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        // Creates a new RNG, adds one to the invocation count.
        SplittableRandom localRNG;
        TrainerProvenance trainerProvenance;
        synchronized(this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }

        ImmutableFeatureMap featureIDs = examples.getFeatureIDMap();
        ImmutableOutputInfo<MultiLabel> labelIDs = examples.getOutputIDInfo();

        ClassifierChainTrainer ccTrainer = new ClassifierChainTrainer(innerTrainer,localRNG.nextLong());

        List<Model<MultiLabel>> models = new ArrayList<>(numMembers);

        for (int i = 0; i < numMembers; i++) {
            logger.info("Building chain " + i);
            models.add(ccTrainer.train(examples));
        }

        EnsembleModelProvenance ensembleProvenance = new EnsembleModelProvenance(
                WeightedEnsembleModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(),
                trainerProvenance, runProvenance, ListProvenance.createListProvenance(models));

        return new WeightedEnsembleModel<>("classifier-chain-ensemble",ensembleProvenance,featureIDs,labelIDs,models,new MultiLabelVotingCombiner());
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
