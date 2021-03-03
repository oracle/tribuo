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

package org.tribuo.sequence;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Trains a sequence model by training a regular model to independently predict every example in each sequence.
 * @param <T> The output type.
 */
public class IndependentSequenceTrainer<T extends Output<T>> implements SequenceTrainer<T> {
    private static final Logger logger = Logger.getLogger(IndependentSequenceTrainer.class.getName());

    @Config(mandatory = true, description = "The trainer to use.")
    private Trainer<T> innerTrainer;

    private int trainInvocationCounter;

    /**
     * Builds a sequence trainer which uses a {@link Trainer} to independently predict each sequence element.
     * @param innerTrainer The trainer to use.
     */
    public IndependentSequenceTrainer(Trainer<T> innerTrainer) {
        this.innerTrainer = innerTrainer;
    }

    /**
     * For olcut.
     */
    private IndependentSequenceTrainer() { }

    @Override
    public IndependentSequenceModel<T> train(SequenceDataset<T> sequenceExamples, Map<String, Provenance> runProvenance) {
        if (sequenceExamples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // Generates the provenance and increments the trainInvocationCounter.
        TrainerProvenance trainerProvenance;
        synchronized (this) {
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        Dataset<T> flatDataset = sequenceExamples.getFlatDataset();

        logger.info(String.format("Training inner trainer with %d examples", flatDataset.size()));
        Model<T> model = innerTrainer.train(flatDataset);

        ModelProvenance provenance = new ModelProvenance(IndependentSequenceModel.class.getName(), OffsetDateTime.now(), sequenceExamples.getProvenance(), trainerProvenance, runProvenance);
        return new IndependentSequenceModel<>("independent-sequence-model", provenance, model);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public String toString() {
        return "IndependentSequenceTrainer(innerTrainer=" + innerTrainer.toString() + ")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
