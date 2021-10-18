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

package org.tribuo.multilabel.baseline;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.hash.HashedFeatureMap;
import org.tribuo.multilabel.ImmutableMultiLabelInfo;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Map;

/**
 * Trains n independent binary {@link Model}s, each of which predicts a single {@link Label}.
 * <p>
 * Then wraps it up in an {@link IndependentMultiLabelModel} to provide a {@link MultiLabel}
 * prediction.
 * <p>
 * It trains each model sequentially, and could be optimised to train in parallel.
 * <p>
 * This trainer implements the approach known as "Binary Relevance" in
 * the multi-label classification literature.
 */
public class IndependentMultiLabelTrainer implements Trainer<MultiLabel> {

    @Config(mandatory = true,description="Trainer to use for each individual label.")
    private Trainer<Label> innerTrainer;

    private int trainInvocationCounter = 0;

    /**
     * for olcut.
     */
    private IndependentMultiLabelTrainer() {}

    /**
     * Constructs an independent multi-label trainer wrapped around the supplied classification trainer.
     * @param innerTrainer The trainer to use for each individual label.
     */
    public IndependentMultiLabelTrainer(Trainer<Label> innerTrainer) {
        this.innerTrainer = innerTrainer;
    }

    @Override
    public Model<MultiLabel> train(Dataset<MultiLabel> examples, Map<String, Provenance> runProvenance) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        ImmutableMultiLabelInfo labelInfo = (ImmutableMultiLabelInfo) examples.getOutputIDInfo();
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        if (featureMap instanceof HashedFeatureMap) {
            throw new IllegalStateException("Cannot use HashingTrainer wrapped around IndependentMultiLabelTrainer.");
        }
        ArrayList<Model<Label>> modelsList = new ArrayList<>();
        ArrayList<Label> labelList = new ArrayList<>();

        // Build provenance
        DatasetProvenance datasetProvenance = examples.getProvenance();
        TrainerProvenance trainerProvenance;
        // Construct the trainer provenance including the inner trainer invocation count field
        synchronized (innerTrainer) {
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        //TODO supply more suitable provenance showing there are multiple models, one per dimension.
        ModelProvenance provenance = new ModelProvenance(IndependentMultiLabelModel.class.getName(), OffsetDateTime.now(), datasetProvenance, trainerProvenance, runProvenance);

        // Construct binarised training data
        MutableDataset<Label> trainingData = new MutableDataset<>(datasetProvenance, new LabelFactory());
        for (Example<MultiLabel> e : examples) {
            trainingData.add(new BinaryExample(e, MultiLabel.NEGATIVE_LABEL));
        }
        for (MultiLabel l : labelInfo.getDomain()) {
            Label label = new Label(l.getLabelString());
            labelList.add(label);
            for (int i = 0; i < examples.size(); i++) {
                Example<MultiLabel> e = examples.getExample(i);
                BinaryExample be = (BinaryExample) trainingData.getExample(i);
                Label newLabel = e.getOutput().createLabel(label);
                // This sets the label in the binary example to either label or MultiLabel.NEGATIVE_LABEL_STRING.
                be.setLabel(newLabel);
            }
            trainingData.regenerateOutputInfo();
            modelsList.add(innerTrainer.train(trainingData));
        }
        return new IndependentMultiLabelModel(labelList,modelsList,provenance,featureMap,labelInfo);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public String toString() {
        return "IndependentMultiLabelTrainer(innerTrainer="+innerTrainer.toString()+")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}

