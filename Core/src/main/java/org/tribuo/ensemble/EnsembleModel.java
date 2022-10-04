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
import org.tribuo.provenance.EnsembleModelProvenance;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.PriorityQueue;

/**
 * A model which contains a list of other {@link Model}s.
 */
public abstract class EnsembleModel<T extends Output<T>> extends Model<T> {
    private static final long serialVersionUID = 1L;

    /**
     * The list of models in this ensemble.
     */
    protected final List<Model<T>> models;

    /**
     * Builds an EnsembleModel from the supplied model list.
     * @param name The name of this ensemble.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param newModels The ensemble members.
     */
    protected EnsembleModel(String name, EnsembleModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, List<Model<T>> newModels) {
        super(name,provenance,featureIDMap,outputIDInfo,true);
        models = Collections.unmodifiableList(newModels);
    }

    /**
     * Returns an unmodifiable view on the ensemble members.
     * @return The ensemble members.
     */
    public List<Model<T>> getModels() {
        return models;
    }

    /**
     * The number of ensemble members.
     * @return The ensemble size.
     */
    public int getNumModels() {
        return models.size();
    }

    @Override
    public abstract Optional<Excuse<T>> getExcuse(Example<T> example);

    @Override
    public EnsembleModelProvenance getProvenance() {
        return (EnsembleModelProvenance) provenance;
    }

    @Override
    protected Model<T> copy(String name, ModelProvenance newProvenance) {
        return copy(name,(EnsembleModelProvenance)newProvenance,new ArrayList<>(models));
    }

    /**
     * Copies this ensemble model.
     * @param name The new name.
     * @param newProvenance The new provenance.
     * @param newModels The new models.
     * @return A copy of the ensemble model.
     */
    protected abstract EnsembleModel<T> copy(String name, EnsembleModelProvenance newProvenance, List<Model<T>> newModels);

    @Override
    public Map<String, List<Pair<String,Double>>> getTopFeatures(int n) {
        Map<String, Map<String,Pair<String,Double>>> featureMap = new HashMap<>();
        for (Model<T> model : models) {
            Map<String, List<Pair<String,Double>>> scoredFeatures = model.getTopFeatures(n);
            for (Entry<String,List<Pair<String,Double>>> e : scoredFeatures.entrySet()) {
                Map<String, Pair<String, Double>> curSet = featureMap.computeIfAbsent(e.getKey(), k -> new HashMap<>());
                for (Pair<String,Double> f : e.getValue()) {
                    Pair<String,Double> tmp = new Pair<>(f.getA(),f.getB()/models.size());
                    curSet.merge(tmp.getA(),tmp,(Pair<String,Double> p1, Pair<String,Double> p2) -> new Pair<>(p1.getA(),p1.getB()+p2.getB()) );
                }
            }
        }

        int maxFeatures = n < 0 ? featureIDMap.size() : n;

        Comparator<Pair<String,Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));
        Map<String, List<Pair<String,Double>>> map = new HashMap<>();
        for (Entry<String, Map<String, Pair<String,Double>>> e : featureMap.entrySet()) {

            PriorityQueue<Pair<String,Double>> q = new PriorityQueue<>(maxFeatures, comparator);
            for (Pair<String,Double> cur : e.getValue().values()) {
                if (q.size() < maxFeatures) {
                    q.offer(cur);
                } else if (comparator.compare(cur, q.peek()) > 0) {
                    q.poll();
                    q.offer(cur);
                }
            }
            List<Pair<String,Double>> list = new ArrayList<>();
            while (q.size() > 0) {
                list.add(q.poll());
            }
            Collections.reverse(list);
            map.put(e.getKey(), list);
        }

        return map;
    }

}
