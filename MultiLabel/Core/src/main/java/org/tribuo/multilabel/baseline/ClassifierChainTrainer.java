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

package org.tribuo.multilabel.baseline;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
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
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A trainer for a Classifier Chain.
 * <p>
 * Classifier chains convert binary classifiers into multi-label
 * classifiers by training one classifier per label (similar to
 * the Binary Relevance approach), but in a specific order (the chain).
 * Classifiers further down the chain use the labels from all previously
 * computed classifiers as features, thus allowing the model to incorporate
 * some measure of label dependence.
 * <p>
 * Choosing the optimal label ordering is tricky as the label dependence
 * is usually unknown, so one popular alternative is to produce an ensemble
 * of randomly ordered chains, which mitigates a poor label ordering by averaging
 * across many orderings.
 * <p>
 * See:
 * <pre>
 * Read, J., Pfahringer, B., Holmes, G., & Frank, E.
 * "Classifier Chains for Multi-Label Classification"
 * Machine Learning, pages 333-359, 2011.
 * </pre>
 */
public final class ClassifierChainTrainer implements Trainer<MultiLabel> {
    private static final Logger logger = Logger.getLogger(ClassifierChainTrainer.class.getName());

    public static final String CC_PREFIX = "CC_FEATURES";
    public static final String CC_POSITIVE = "POSITIVE";
    public static final String CC_NEGATIVE = "NEGATIVE";
    public static final String CC_SEPARATOR = "_";

    @Config(mandatory = true, description = "The trainer to use.")
    private Trainer<Label> innerTrainer;

    @Config(mandatory = false, description = "Label order.")
    private List<String> labelOrder = Collections.emptyList();

    @Config(mandatory = false, description = "Randomise the label chain order.")
    private boolean randomOrder = false;

    @Config(mandatory = false, description = "RNG seed for random label orders.")
    private long seed = Trainer.DEFAULT_SEED;

    private int trainInvocationCounter = 0;
    private SplittableRandom rng;

    /**
     * For OLCUT.
     */
    private ClassifierChainTrainer() {}

    public ClassifierChainTrainer(Trainer<Label> innerTrainer, long seed) {
        this.innerTrainer = innerTrainer;
        this.labelOrder = Collections.emptyList();
        this.randomOrder = true;
        this.seed = seed;
        postConfig();
    }

    public ClassifierChainTrainer(Trainer<Label> innerTrainer, List<String> labelOrder) {
        this.innerTrainer = innerTrainer;
        this.labelOrder = Collections.unmodifiableList(new ArrayList<>(labelOrder));
        this.randomOrder = false;
        this.seed = Trainer.DEFAULT_SEED;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (!randomOrder && labelOrder.isEmpty()) {
            throw new PropertyException("","randomOrder","Either randomOrder must be true, or labelOrder must be non-empty");
        }
        this.rng = new SplittableRandom(this.seed);
    }

    @Override
    public Model<MultiLabel> train(Dataset<MultiLabel> examples, Map<String, Provenance> runProvenance) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // Creates a new RNG, adds one to the invocation count, generates the provenance.
        TrainerProvenance trainerProvenance;
        SplittableRandom localRNG;
        synchronized(this) {
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableMultiLabelInfo labelInfo = (ImmutableMultiLabelInfo) examples.getOutputIDInfo();
        Set<MultiLabel> domain = labelInfo.getDomain();
        ArrayList<Label> curLabelOrder;
        if (randomOrder) {
            // generate random label ordering
            curLabelOrder = new ArrayList<>(domain.size());
            for (MultiLabel m : domain) {
                curLabelOrder.add(new Label(m.getLabelString()));
            }
            Util.shuffle(curLabelOrder,localRNG);
        } else {
            if (labelInfo.size() != labelOrder.size()) {
                throw new IllegalArgumentException("Must supply a total label ordering, labelOrder = " + labelOrder.toString() + ", train label domain = " + labelInfo.getDomain());
            } else {
                // validate all the labels exist
                for (String lbl : labelOrder) {
                    if (labelInfo.getLabelCount(lbl) == 0) {
                        throw new IllegalArgumentException("Must supply a total label ordering, labelOrder = " + labelOrder.toString() + ", train label domain = " + labelInfo.getDomain());
                    }
                }
            }
            curLabelOrder = new ArrayList<>(labelOrder.size());
            for (String s : labelOrder) {
                curLabelOrder.add(new Label(s));
            }
        }
        logger.log(Level.INFO, "Training with label order " + curLabelOrder);
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        if (featureMap instanceof HashedFeatureMap) {
            throw new IllegalStateException("Cannot use HashingTrainer wrapped around ClassifierChainTrainer.");
        }
        ArrayList<Model<Label>> modelsList = new ArrayList<>();
        DatasetProvenance datasetProvenance = examples.getProvenance();
        MutableDataset<Label> trainingData = new MutableDataset<>(datasetProvenance, new LabelFactory());
        Label curLabel = curLabelOrder.get(0);
        for (Example<MultiLabel> e : examples) {
            Label newLabel = e.getOutput().createLabel(curLabel);
            // This sets the label in the new example to either curLabel or MultiLabel.NEGATIVE_LABEL_STRING.
            trainingData.add(new BinaryExample(e, newLabel));
        }
        for (int i = 0; i < curLabelOrder.size(); i++) {
            modelsList.add(innerTrainer.train(trainingData));
            if (i != (curLabelOrder.size() - 1)) {
                // if it's not the last iteration, add the current labels as features to the dataset.
                Label nextLabel = curLabelOrder.get(i+1);
                for (int j = 0; j < trainingData.size(); j++) {
                    BinaryExample curExample = (BinaryExample) trainingData.getExample(j);

                    // Add the current label as a feature
                    Label oldLabel = curExample.getOutput();
                    String oldLabelName = curLabelOrder.get(i).getLabel();
                    String featureName;
                    if (oldLabel == MultiLabel.NEGATIVE_LABEL) {
                        featureName = CC_PREFIX + CC_SEPARATOR + oldLabelName + CC_SEPARATOR + CC_NEGATIVE;
                    } else {
                        featureName = CC_PREFIX + CC_SEPARATOR + oldLabelName + CC_SEPARATOR + CC_POSITIVE;
                    }
                    curExample.add(new Feature(featureName,1.0));

                    // Update the label for the next iteration
                    Label newLabel = examples.getExample(j).getOutput().createLabel(nextLabel);
                    curExample.setLabel(newLabel);
                }
                // Update the domains with the new features & outputs
                trainingData.regenerateOutputInfo();
                trainingData.regenerateFeatureInfo();
            }
        }
        ModelProvenance provenance = new ModelProvenance(ClassifierChainModel.class.getName(), OffsetDateTime.now(), datasetProvenance, trainerProvenance, runProvenance);
        return new ClassifierChainModel(Collections.unmodifiableList(curLabelOrder),modelsList,provenance,featureMap,labelInfo);
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
