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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.provenance.EnsembleModelProvenance;
import org.tribuo.util.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;

/**
 * An ensemble model that uses weights to combine the ensemble member predictions.
 */
public final class WeightedEnsembleModel<T extends Output<T>> extends EnsembleModel<T> {
    private static final long serialVersionUID = 1L;

    protected final float[] weights;

    protected final EnsembleCombiner<T> combiner;

    public WeightedEnsembleModel(String name, EnsembleModelProvenance description, ImmutableFeatureMap featureIDMap,
                                 ImmutableOutputInfo<T> outputIDInfo,
                                 List<Model<T>> newModels, EnsembleCombiner<T> combiner) {
        this(name,description,featureIDMap,outputIDInfo,newModels, combiner, Util.generateUniformVector(newModels.size(), 1.0f/newModels.size()));
    }

    public WeightedEnsembleModel(String name, EnsembleModelProvenance description, ImmutableFeatureMap featureIDMap,
                          ImmutableOutputInfo<T> outputIDInfo,
                          List<Model<T>> newModels, EnsembleCombiner<T> combiner, float[] weights) {
        super(name,description,featureIDMap,outputIDInfo,newModels);
        this.weights = Arrays.copyOf(weights,weights.length);
        this.combiner = combiner;
    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        List<Prediction<T>> predictions = new ArrayList<>();
        for (Model<T> model : models) {
            predictions.add(model.predict(example));
        }

        return combiner.combine(outputIDInfo,predictions,weights);
    }

    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        Map<String, Map<String,Double>> map = new HashMap<>();
        Prediction<T> prediction = predict(example);
        List<Excuse<T>> excuses = new ArrayList<>();

        for (int i = 0; i < models.size(); i++) {
            Optional<Excuse<T>> excuse = models.get(i).getExcuse(example);
            if (excuse.isPresent()) {
                excuses.add(excuse.get());
                Map<String, List<Pair<String,Double>>> m = excuse.get().getScores();
                for (Entry<String, List<Pair<String,Double>>> e : m.entrySet()) {
                    Map<String, Double> innerMap = map.computeIfAbsent(e.getKey(), k -> new HashMap<>());
                    for (Pair<String,Double> p : e.getValue()) {
                        innerMap.merge(p.getA(), p.getB() * weights[i], Double::sum);
                    }
                }
            }
        }

        if (map.isEmpty()) {
            return Optional.empty();
        } else {
            Map<String, List<Pair<String, Double>>> outputMap = new HashMap<>();
            for (Entry<String, Map<String, Double>> label : map.entrySet()) {
                List<Pair<String, Double>> list = new ArrayList<>();

                for (Entry<String, Double> entry : label.getValue().entrySet()) {
                    list.add(new Pair<>(entry.getKey(), entry.getValue()));
                }

                list.sort((Pair<String, Double> o1, Pair<String, Double> o2) -> o2.getB().compareTo(o1.getB()));
                outputMap.put(label.getKey(), list);
            }

            return Optional.of(new EnsembleExcuse<>(example, prediction, outputMap, excuses));
        }
    }

    @Override
    protected EnsembleModel<T> copy(String name, EnsembleModelProvenance newProvenance, List<Model<T>> newModels) {
        return new WeightedEnsembleModel<>(name,newProvenance,featureIDMap,outputIDInfo,newModels,combiner);
    }
}
