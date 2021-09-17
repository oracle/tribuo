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

package org.tribuo.classification.sequence.viterbi;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.sequence.viterbi.ViterbiModel.ScoreAggregation;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.sequence.ImmutableSequenceDataset;
import org.tribuo.sequence.MutableSequenceDataset;
import org.tribuo.sequence.SequenceDataset;
import org.tribuo.sequence.SequenceExample;
import org.tribuo.sequence.SequenceModel;
import org.tribuo.sequence.SequenceTrainer;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Builds a Viterbi model using the supplied {@link Trainer}.
 * Has a parameter to control the label features which are added to the features supplied by the data.
 */
public final class ViterbiTrainer implements SequenceTrainer<Label> {

    @Config(mandatory = true, description = "Inner trainer for each sequence element.")
    private Trainer<Label> trainer;

    @Config(mandatory = true, description = "Feature extractor to pull in surrounding label features.")
    private LabelFeatureExtractor labelFeatureExtractor;

    @Config(mandatory = true, description = "Number of candidate paths.")
    private int stackSize;

    @Config(mandatory = true, description = "Score aggregation function.")
    private ScoreAggregation scoreAggregation;

    private int trainInvocationCounter = 0;

    /**
     * Constructs a ViterbiTrainer wrapping the specified trainer, with an unbounded stack size.
     * @param trainer The trainer to wrap.
     * @param labelFeatureExtractor The feature extraction function for labels.
     * @param scoreAggregation The score aggregation function.
     */
    public ViterbiTrainer(Trainer<Label> trainer, LabelFeatureExtractor labelFeatureExtractor,
                          ScoreAggregation scoreAggregation) {
        this(trainer, labelFeatureExtractor, -1, scoreAggregation);
    }

    /**
     * Constructs a ViterbiTrainer wrapping the specified trainer.
     * @param trainer The trainer to wrap.
     * @param labelFeatureExtractor The feature extraction function for labels.
     * @param stackSize The stack size.
     * @param scoreAggregation The score aggregation function.
     */
    public ViterbiTrainer(Trainer<Label> trainer, LabelFeatureExtractor labelFeatureExtractor, int stackSize,
                          ScoreAggregation scoreAggregation) {
        this.trainer = trainer;
        this.labelFeatureExtractor = labelFeatureExtractor;
        this.stackSize = stackSize;
        this.scoreAggregation = scoreAggregation;
    }

    /**
     * For OLCUT.
     */
    private ViterbiTrainer() { }

    /**
     * The viterbi train method is unique because it delegates to a regular
     * {@link Model} train method, but before it does, it adds features derived
     * from preceding labels. The pipeline upstream of this call should not care
     * that these features are being added - that is, we would not want to make
     * the upstream logic worry about what kind of trainer will be used and have
     * conditional logic that says to add special label-derived features if
     * using the ViterbiTrainer. So, these one-of-a-kind unique-in-the-world
     * label-derived features are generated here and added to the sequence
     * examples of the passed in dataset. If you pass in a
     * MutableSequenceDataset, then please be aware that your dataset will be
     * modified after calling this method and therefore subsequent calls to
     * other SequenceModel.train methods with your dataset should be avoided. If
     * you pass in an ImmutableSequenceDataset, then please be aware that your
     * entire dataset is going to be copied as a MutableSequenceDataset - so
     * there is a memory penalty.
     * @param dataset The input dataset.
     * @param runProvenance Any additional information to record in the provenance.
     * @return A {@link SequenceModel} using Viterbi wrapped around an inner {@link Model}.
     */
    @Override
    public SequenceModel<Label> train(SequenceDataset<Label> dataset, Map<String, Provenance> runProvenance) {
        if (dataset.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // if stack size isn't specified, then we will calculate it based on the
        // number of unique output values
        if (stackSize == -1) {
            stackSize = dataset.getOutputIDInfo().size();
        }

        // create a copy of the dataset to a mutable one. See note above.
        if (dataset instanceof ImmutableSequenceDataset) {
            dataset = new MutableSequenceDataset<>((ImmutableSequenceDataset<Label>) dataset);
        }

        if (!(dataset instanceof MutableSequenceDataset)) {
            throw new IllegalArgumentException("unable to handle sub-type of dataset: " + dataset.getClass().getName());
        }

        for (SequenceExample<Label> sequenceExample : dataset) {
            List<Label> labels = new ArrayList<>();

            for (Example<Label> example : sequenceExample) {
                List<Feature> labelFeatures = extractFeatures(labels, (MutableSequenceDataset<Label>) dataset,
                        1.0);
                example.addAll(labelFeatures);
                labels.add(example.getOutput());
            }
        }

        TrainerProvenance trainerProvenance = getProvenance();
        ModelProvenance provenance = new ModelProvenance(ViterbiModel.class.getName(), OffsetDateTime.now(), dataset.getProvenance(), trainerProvenance, runProvenance);
        trainInvocationCounter++;
        Dataset<Label> flatData = dataset.getFlatDataset();
        Model<Label> model = trainer.train(flatData);
        return new ViterbiModel("viterbi+" + model.getName(), provenance, model,
                labelFeatureExtractor, stackSize, scoreAggregation);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    private List<Feature> extractFeatures(List<Label> labels,
                                          MutableSequenceDataset<Label> dataset, double value) {
        List<Feature> labelFeatures = new ArrayList<>();
        for (Feature labelFeature : labelFeatureExtractor.extractFeatures(labels, value)) {
            dataset.getFeatureMap().add(labelFeature.getName(), labelFeature.getValue());
            labelFeatures.add(labelFeature);
        }
        return labelFeatures;
    }

    @Override
    public String toString() {
        return "ViterbiTrainer(innerTrainer=" + trainer.toString() + ",labelFeatureExtractor=" + labelFeatureExtractor.toString() + ")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
