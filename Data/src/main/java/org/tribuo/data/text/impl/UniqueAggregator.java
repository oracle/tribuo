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

package org.tribuo.data.text.impl;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Feature;
import org.tribuo.data.text.FeatureAggregator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Aggregates feature tokens, generating unique features.
 */
public class UniqueAggregator implements FeatureAggregator {

    private final ThreadLocal<Map<String,Double>> map = ThreadLocal.withInitial(HashMap::new);

    @Config(description="Value to emit, if unset emits the last value observed for that token.")
    private double value = Double.NaN;

    /**
     * Constructs an aggregator that replaces all features with the same
     * name with a single feature with the specified value.
     * @param value The inserted feature value.
     */
    public UniqueAggregator(double value) {
        this.value = value;
    }

    /**
     * Constructs an aggregator that replaces all features with the same
     * name with a single feature with the last observed value of that feature.
     */
    public UniqueAggregator() { }

    @Override
    public List<Feature> aggregate(List<Feature> input) {
        Map<String,Double> curMap = map.get();
        curMap.clear();

        for (Feature f : input) {
            curMap.put(f.getName(),f.getValue());
        }

        List<Feature> features = new ArrayList<>();

        for (Map.Entry<String,Double> e : curMap.entrySet()) {
            double tmpValue;
            if (Double.isNaN(value)) {
                tmpValue = e.getValue();
            } else {
                tmpValue = value;
            }
            features.add(new Feature(e.getKey(),tmpValue));
        }

        return features;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FeatureAggregator");
    }
}
