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

package org.tribuo.hash;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.ImmutableDataset;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A {@link Trainer} which hashes the {@link Dataset} before the {@link Model}
 * is produced. This means the model does not contain any feature names,
 * only one way hashes of names.
 * <p>
 * It wraps another Trainer which actually performs the training.
 * @param <T> The type of Output this trainer works with.
 */
public final class HashingTrainer<T extends Output<T>> implements Trainer<T> {
    private static final Logger logger = Logger.getLogger(HashingTrainer.class.getName());

    @Config(mandatory = true,description="Trainer to use.")
    private Trainer<T> innerTrainer;

    @Config(mandatory = true,description="Feature hashing function to use.")
    private Hasher hasher;

    /**
     * For olcut.
     */
    private HashingTrainer() {}

    /**
     * Constructs a hashing trainer using the supplied parameters.
     * @param trainer The trainer to use.
     * @param hasher The feature hasher to apply.
     */
    public HashingTrainer(Trainer<T> trainer, Hasher hasher) {
        this.innerTrainer = trainer;
        this.hasher = hasher;
    }

    /**
     * This clones the {@link Dataset}, hashes each of the examples
     * and rewrites their feature ids before passing it to the inner trainer.
     * <p>
     * This ensures the Trainer sees the data after the collisions, and thus
     * builds the correct size data structures.
     * @param dataset The input dataset.
     * @param instanceProvenance Provenance information specific to this execution of train (e.g., cross validation fold number).
     * @return A trained {@link Model}.
     */
    @Override
    public Model<T> train(Dataset<T> dataset,Map<String, Provenance> instanceProvenance) {
        return(train(dataset, instanceProvenance, INCREMENT_INVOCATION_COUNT));
    }

    public Model<T> train(Dataset<T> dataset,Map<String, Provenance> instanceProvenance, int invocationCount) {
        logger.log(Level.INFO,"Before hashing, had " + dataset.getFeatureMap().size() + " features.");
        ImmutableDataset<T> hashedData = ImmutableDataset.hashFeatureMap(dataset, hasher);
        logger.log(Level.INFO,"After hashing, had " + hashedData.getFeatureMap().size() + " features.");
        Model<T> model = innerTrainer.train(hashedData,instanceProvenance, invocationCount);
        if (!(model.getFeatureIDMap() instanceof HashedFeatureMap)) {
            //
            // This exception is thrown when the innerTrainer did not copy the ImmutableFeatureMap from the
            // ImmutableDataset, but modified it in some way. For example Viterbi will do this.
            throw new IllegalStateException("Trainer " + innerTrainer.getClass().getName() + " does not support hashing.");
        }
        return model;
    }

    @Override
    public int getInvocationCount() {
        return innerTrainer.getInvocationCount();
    }

    @Override
    public synchronized void setInvocationCount(int invocationCount){
        innerTrainer.setInvocationCount(invocationCount);
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
