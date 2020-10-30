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

package org.tribuo;

import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A model which uses a subset of the features it knows about to make predictions.
 */
public abstract class SparseModel<T extends Output<T>> extends Model<T> {
    private static final long serialVersionUID = 1L;

    private final Map<String,List<String>> activeFeatures;

    /**
     * Constructs a sparse model from the supplied arguments.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The features the model knows.
     * @param outputIDInfo The outputs the model can produce.
     * @param generatesProbabilities Does this model generate probabilistic outputs.
     * @param activeFeatures The active features in this model.
     */
    public SparseModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, boolean generatesProbabilities, Map<String,List<String>> activeFeatures) {
        super(name, provenance, featureIDMap, outputIDInfo, generatesProbabilities);
        Map<String,List<String>> tmpActiveFeatures = new HashMap<>();
        for (Map.Entry<String,List<String>> e : activeFeatures.entrySet()) {
            List<String> features = new ArrayList<>(e.getValue());
            Collections.sort(features);
            tmpActiveFeatures.put(e.getKey(),Collections.unmodifiableList(features));
        }
        this.activeFeatures = Collections.unmodifiableMap(tmpActiveFeatures);
    }

    /**
     * Return an immutable view on the active features for each dimension.
     * <p>
     * Sorted lexicographically.
     * @return The active features.
     */
    public Map<String,List<String>> getActiveFeatures() {
        return activeFeatures;
    }

    @Override
    public SparseModel<T> copy() {
        return (SparseModel<T>) super.copy();
    }

}
