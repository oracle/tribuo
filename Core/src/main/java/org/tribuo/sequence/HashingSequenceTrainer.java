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

package org.tribuo.sequence;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Output;
import org.tribuo.hash.HashedFeatureMap;
import org.tribuo.hash.Hasher;
import org.tribuo.provenance.SkeletalTrainerProvenance;
import org.tribuo.provenance.TrainerProvenance;

import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A SequenceTrainer that hashes all the feature names on the way in.
 * <p>
 * It wraps another SequenceTrainer which actually builds the {@link SequenceModel}.
 * @param <T> The type of the output.
 */
public final class HashingSequenceTrainer<T extends Output<T>> implements SequenceTrainer<T> {
    private static final Logger logger = Logger.getLogger(HashingSequenceTrainer.class.getName());

    @Config(mandatory = true,description="Trainer to use.")
    private SequenceTrainer<T> innerTrainer;

    @Config(mandatory = true,description="Feature hashing function to use.")
    private Hasher hasher;

    /**
     * For olcut.
     */
    private HashingSequenceTrainer() {}

    /**
     * Constructs a hashing sequence trainer using the supplied parameters.
     * @param trainer The sequence trainer to use.
     * @param hasher The feature hasher to apply.
     */
    public HashingSequenceTrainer(SequenceTrainer<T> trainer, Hasher hasher) {
        this.innerTrainer = trainer;
        this.hasher = hasher;
    }

    /**
     * This clones the {@link SequenceDataset}, hashes each of the examples
     * and rewrites their feature ids before passing it to the inner trainer.
     * <p>
     * This ensures the Trainer sees the data after the collisions, and thus
     * builds the correct size data structures.
     * @param sequenceExamples The input dataset.
     * @param instanceProvenance Training run specific provenance information.
     * @return A trained {@link SequenceModel}.
     */
    @Override
    public SequenceModel<T> train(SequenceDataset<T> sequenceExamples, Map<String, Provenance> instanceProvenance) {
        logger.log(Level.INFO,"Before hashing, had " + sequenceExamples.getFeatureIDMap().size() + " features.");
        SequenceDataset<T> hashedData = ImmutableSequenceDataset.changeFeatureMap(sequenceExamples, HashedFeatureMap.generateHashedFeatureMap(sequenceExamples.getFeatureIDMap(),hasher));
        logger.log(Level.INFO,"After hashing, had " + hashedData.getFeatureIDMap().size() + " features.");
        SequenceModel<T> model = innerTrainer.train(hashedData,instanceProvenance);
        if (!(model.featureIDMap instanceof HashedFeatureMap)) {
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
    public String toString() {
        return "HashingSequenceTrainer(trainer="+innerTrainer.toString()+",hasher="+hasher.toString()+")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new HashingSequenceTrainerProvenance(this);
    }

    /**
     * Provenance for {@link HashingSequenceTrainer}.
     */
    public static class HashingSequenceTrainerProvenance extends SkeletalTrainerProvenance {
        private static final long serialVersionUID = 1L;

        <T extends Output<T>> HashingSequenceTrainerProvenance(HashingSequenceTrainer<T> host) {
            super(host);
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public HashingSequenceTrainerProvenance(Map<String, Provenance> map) {
            super(extractProvenanceInfo(map));
        }
    }
}
