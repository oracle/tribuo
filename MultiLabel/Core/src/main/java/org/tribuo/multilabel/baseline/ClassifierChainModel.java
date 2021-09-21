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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.tribuo.multilabel.baseline.ClassifierChainTrainer.CC_NEGATIVE;
import static org.tribuo.multilabel.baseline.ClassifierChainTrainer.CC_POSITIVE;
import static org.tribuo.multilabel.baseline.ClassifierChainTrainer.CC_PREFIX;
import static org.tribuo.multilabel.baseline.ClassifierChainTrainer.CC_SEPARATOR;

/**
 * A Classifier Chain Model.
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
 * Read, J., Pfahringer, B., Holmes, G., &amp; Frank, E.
 * "Classifier Chains for Multi-Label Classification"
 * Machine Learning, pages 333-359, 2011.
 * </pre>
 */
public final class ClassifierChainModel extends Model<MultiLabel> {
    private static final long serialVersionUID = 1L;

    private final List<Model<Label>> models;
    private final List<Label> labelOrder;

    /**
     * The list of Label and list of Model must be in the same order, and have a bijection.
     * @param labelOrder The list of labels this model was trained on.
     * @param models The list of individual binary models.
     * @param description A description of the trainer.
     * @param featureMap The feature domain used in training.
     * @param labelInfo The label domain used in training.
     */
    ClassifierChainModel(List<Label> labelOrder, List<Model<Label>> models, ModelProvenance description, ImmutableFeatureMap featureMap, ImmutableOutputInfo<MultiLabel> labelInfo) {
        super("classifier-chain",description,featureMap,labelInfo,false);
        this.labelOrder = labelOrder;
        this.models = models;
    }

    @Override
    public Prediction<MultiLabel> predict(Example<MultiLabel> example) {
        Set<Label> predictedLabels = new HashSet<>();
        BinaryExample e = new BinaryExample(example,MultiLabel.NEGATIVE_LABEL);
        int numUsed = 0;
        for (int i = 0; i < labelOrder.size(); i++) {
            Model<Label> curModel = models.get(i);
            Label curLabel = labelOrder.get(i);
            Prediction<Label> p = curModel.predict(e);
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            String featureName;
            if (!p.getOutput().getLabel().equals(MultiLabel.NEGATIVE_LABEL_STRING)) {
                predictedLabels.add(p.getOutput());
                // update example with positive label feature.
                featureName = CC_PREFIX + CC_SEPARATOR + curLabel.getLabel() + CC_SEPARATOR + CC_POSITIVE;
            } else {
                // update example with negative label feature.
                featureName = CC_PREFIX + CC_SEPARATOR + curLabel.getLabel() + CC_SEPARATOR + CC_NEGATIVE;
            }
            e.add(new Feature(featureName,1.0));
        }
        return new Prediction<>(new MultiLabel(predictedLabels),numUsed,example);
    }

    /**
     * Returns the training label order.
     * @return The training label order.
     */
    public List<Label> getLabelOrder() {
        return labelOrder;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    public Optional<Excuse<MultiLabel>> getExcuse(Example<MultiLabel> example) {
        return Optional.empty();
    }

    @Override
    protected ClassifierChainModel copy(String newName, ModelProvenance newProvenance) {
        List<Model<Label>> newModels = new ArrayList<>();
        for (Model<Label> e : models) {
            newModels.add(e.copy());
        }
        return new ClassifierChainModel(labelOrder,newModels,newProvenance,featureIDMap,outputIDInfo);
    }
}
