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

package org.tribuo.multilabel.baseline;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * A {@link Model} which wraps n binary models, where n is the
 * size of the MultiLabel domain. Each model independently predicts
 * a single binary label.
 * <p>
 * It is possible for the prediction to produce an empty MultiLabel
 * when none of the binary Labels were predicted.
 * <p>
 * This model implements the approach known as "Binary Relevance" in
 * the multi-label classification literature.
 */
public class IndependentMultiLabelModel extends Model<MultiLabel> {
    private static final long serialVersionUID = 1L;

    private final List<Model<Label>> models;
    private final List<Label> labels;

    /**
     * The list of Label and list of Model must be in the same order, and have a bijection.
     *
     * @param labels      The list of labels this model was trained on.
     * @param models      The list of individual binary models.
     * @param description A description of the trainer.
     * @param featureMap  The feature domain used in training.
     * @param labelInfo   The label domain used in training.
     */
    IndependentMultiLabelModel(List<Label> labels, List<Model<Label>> models, ModelProvenance description, ImmutableFeatureMap featureMap, ImmutableOutputInfo<MultiLabel> labelInfo) {
        super("binary-relevance", description, featureMap, labelInfo, models.get(0).generatesProbabilities());
        this.labels = labels;
        this.models = models;
    }

    @Override
    public Prediction<MultiLabel> predict(Example<MultiLabel> example) {
        Set<Label> predictedLabels = new HashSet<>();
        BinaryExample e = new BinaryExample(example, null);
        int numUsed = 0;
        for (Model<Label> m : models) {
            Prediction<Label> p = m.predict(e);
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            if (!p.getOutput().getLabel().equals(MultiLabel.NEGATIVE_LABEL_STRING)) {
                predictedLabels.add(p.getOutput());
            }
        }
        return new Prediction<>(new MultiLabel(predictedLabels), numUsed, example);
    }

    /**
     * This aggregates the top features from each of the models.
     * <p>
     * If the individual models support per label features, then only the features
     * for the positive label are aggregated.
     *
     * @param n the number of features to return. If this value is less than 0,
     *          all features should be returned for each class.
     * @return The top n features.
     */
    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        Map<String, List<Pair<String, Double>>> map = new HashMap<>();
        for (int i = 0; i < models.size(); i++) {
            Model<Label> m = models.get(i);
            String label = labels.get(i).getLabel();
            Map<String, List<Pair<String, Double>>> modelMap = m.getTopFeatures(n);
            if (modelMap != null) {
                if (modelMap.size() == 1) {
                    map.put(label, modelMap.get(Model.ALL_OUTPUTS));
                } else {
                    map.merge(label, modelMap.get(label), (List<Pair<String, Double>> l, List<Pair<String, Double>> r) -> {
                        l.addAll(r);
                        return l;
                    });
                }
            }
        }
        return map;
    }

    @Override
    public Optional<Excuse<MultiLabel>> getExcuse(Example<MultiLabel> example) {
        //TODO implement this to return the per label excuses.
        return Optional.empty();
    }

    @Override
    protected IndependentMultiLabelModel copy(String newName, ModelProvenance newProvenance) {
        List<Model<Label>> newModels = new ArrayList<>();
        for (Model<Label> e : models) {
            newModels.add(e.copy());
        }
        return new IndependentMultiLabelModel(labels, newModels, newProvenance, featureIDMap, outputIDInfo);
    }
}
