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

package org.tribuo.data.columnar.processors.feature;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.data.columnar.ColumnarFeature;
import org.tribuo.data.columnar.FeatureProcessor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Processes a feature list, aggregating all the feature values with the same name.
 * <p>
 * The aggregation is user controllable.
 * <p>
 * In most cases this will be unnecessary as the feature names will be unique as they are keyed by the field name,
 * however it's possible to induce collisions via text fields or other mechanisms.
 */
public class UniqueProcessor implements FeatureProcessor {

    /**
     * The type of reduction operation to perform.
     */
    public enum UniqueType {
        /**
         * Select the first feature value in the list.
         */
        FIRST,
        /**
         * Select the last feature value in the list.
         */
        LAST,
        /**
         * Select the maximum feature value in the list.
         */
        MAX,
        /**
         * Select the minimum feature value in the list.
         */
        MIN,
        /**
         * Add together all the feature values. Uses the field names from the first element.
         */
        SUM;
    }

    @Config(mandatory=true,description="The operation to perform.")
    private UniqueType reductionType;

    /**
     * For OLCUT
     */
    private UniqueProcessor() {}

    /**
     * Creates a UniqueProcessor using the specified reduction operation.
     * @param reductionType The reduction operation to perform.
     */
    public UniqueProcessor(UniqueType reductionType) {
        this.reductionType = reductionType;
    }

    @Override
    public List<ColumnarFeature> process(List<ColumnarFeature> features) {
        if (features.isEmpty()) {
            return features;
        }
        Map<String,List<ColumnarFeature>> map = new LinkedHashMap<>();
        for (ColumnarFeature f : features) {
            map.computeIfAbsent(f.getName(), (s) -> new ArrayList<>()).add(f);
        }

        // Unique the features
        List<ColumnarFeature> returnVal = new ArrayList<>();
        for (Map.Entry<String,List<ColumnarFeature>> e : map.entrySet()) {
            returnVal.add(uniqueList(reductionType, e.getValue()));
        }
        return returnVal;
    }

    /**
     * Processes the list returning the unique feature.
     * <p>
     * Throws {@link IllegalArgumentException} if the list is empty.
     * @param type The unique operation to perform.
     * @param list The list of features to process.
     * @return The unique feature.
     */
    private static ColumnarFeature uniqueList(UniqueType type, List<ColumnarFeature> list) {
        if (list.isEmpty()) {
            throw new IllegalArgumentException("List must contain at least one feature");
        } else if (list.size() == 1) {
            return list.get(0);
        } else {
            switch (type) {
                case FIRST:
                    return list.get(0);
                case LAST:
                    return list.get(list.size()-1);
                case MAX:
                    return list.stream().max(Comparator.comparingDouble(ColumnarFeature::getValue)).get();
                case MIN:
                    return list.stream().min(Comparator.comparingDouble(ColumnarFeature::getValue)).get();
                case SUM:
                    double value = 0.0;
                    for (ColumnarFeature f : list) {
                        value += f.getValue();
                    }
                    ColumnarFeature first = list.get(0);
                    if (first.getFieldName().equals(ColumnarFeature.CONJUNCTION)) {
                        return new ColumnarFeature(first.getFirstFieldName(),first.getSecondFieldName(),first.getColumnEntry(),value);
                    } else {
                        return new ColumnarFeature(first.getFieldName(),first.getColumnEntry(),value);
                    }
                default:
                    throw new IllegalStateException("Unknown enum type " + type);
            }
        }
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FeatureProcessor");
    }
}
