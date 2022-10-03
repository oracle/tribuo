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

package org.tribuo.transform;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Dataset;
import org.tribuo.FeatureMap;
import org.tribuo.MutableDataset;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * A carrier type for a set of transformations to be applied to a {@link Dataset}.
 * <p>
 * Feature specific transformations are specified using a regex. If multiple
 * regexes match a given feature, then an {@link IllegalArgumentException} is thrown
 * when {@link Dataset#createTransformers(TransformationMap)} is called.
 * <p>
 * Global transformations are applied <em>after</em> all feature specific transformations.
 * <p>
 * Transformations only operate on observed values. To operate on implicit zeros then
 * first call {@link MutableDataset#densify} on the datasets.
 * See {@link org.tribuo.transform} for a more detailed discussion of densify.
 */
public class TransformationMap implements Configurable, Provenancable<ConfiguredObjectProvenance> {

    @Config(mandatory = true,description="Global transformations to apply after the feature specific transforms.")
    private List<Transformation> globalTransformations;

    @Config(description="Feature specific transformations. Accepts regexes for feature names.")
    private Map<String,TransformationList> featureTransformationList = new HashMap<>();

    private final Map<String,List<Transformation>> featureTransformations = new HashMap<>();

    private ConfiguredObjectProvenanceImpl provenance;

    /**
     * For OLCUT.
     */
    private TransformationMap() {}

    /**
     * Creates a transformation map from the supplied global transformation list, and per feature transformations.
     * @param globalTransformations The global transformations.
     * @param featureTransformations The per feature transformations.
     */
    public TransformationMap(List<Transformation> globalTransformations, Map<String,List<Transformation>> featureTransformations) {
        this.globalTransformations = new ArrayList<>(globalTransformations);
        this.featureTransformations.putAll(featureTransformations);

        // Copy values out for provenance
        for (Map.Entry<String,List<Transformation>> e : featureTransformations.entrySet()) {
            featureTransformationList.put(e.getKey(),new TransformationList(e.getValue()));
        }
        
    }

    /**
     * Creates a TransformationMap with only global transformations.
     * @param globalTransformations The global transformations.
     */
    public TransformationMap(List<Transformation> globalTransformations) {
        this(globalTransformations, Collections.emptyMap());
    }

    /**
     * Creates a TransformationMap with only per feature transformations.
     * @param featureTransformations The per feature transformations.
     */
    public TransformationMap(Map<String,List<Transformation>> featureTransformations) {
        this(Collections.emptyList(),featureTransformations);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if(globalTransformations.isEmpty() && featureTransformationList.isEmpty()) {
            throw new PropertyException("TransformationMap", 
                    "Both global transformations and feature transformations can't be empty!");
        }
        
        for (Map.Entry<String,TransformationList> e : featureTransformationList.entrySet()) {
            featureTransformations.put(e.getKey(),e.getValue().list);
        }
    }

    /**
     * Checks that a given transformation set doesn't have conflicts when applied to the supplied featureMap.
     * @param featureMap The featureMap to check.
     * @return True if the transformation set doesn't have conflicts, false otherwise.
     */
    public boolean validateTransformations(FeatureMap featureMap) {
        HashSet<String> featuresWithPatterns = new HashSet<>();
        ArrayList<String> featureNames = new ArrayList<>(featureMap.keySet());
        boolean valid = true;

        // Loop over all regexes
        for (String regex : featureTransformations.keySet()) {
            Pattern p = Pattern.compile(regex);
            // Loop over all features
            for (String s : featureNames) {
                // Check if the pattern matches the feature
                if (p.matcher(s).matches()) {
                    // If it matches, add the feature to the HashSet
                    valid = featuresWithPatterns.add(s);
                    // If it already was present, there are two patterns for the same feature
                    // so the Transformations are invalid.
                    // Bail out.
                    if (!valid) {
                        break;
                    }
                }
            }
            if (!valid) {
                break;
            }
        }

        return valid;
    }

    @Override
    public String toString() {
        return "TransformationMap(featureTransformations="+featureTransformations.toString()+",globalTransformations="+globalTransformations.toString()+")";
    }

    /**
     * Gets the global transformations in this TransformationMap.
     * @return The global transformations
     */
    public List<Transformation> getGlobalTransformations() {
        return globalTransformations;
    }

    /**
     * Gets the map of feature specific transformations.
     * @return The feature specific transformations.
     */
    public Map<String, List<Transformation>> getFeatureTransformations() {
        return featureTransformations;
    }

    @Override
    public synchronized ConfiguredObjectProvenance getProvenance() {
        if (provenance == null) {
            provenance = cacheProvenance();
        }
        return provenance;
    }

    private ConfiguredObjectProvenanceImpl cacheProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"TransformationMap");
    }

    /**
     * A carrier type as OLCUT does not support nested generics.
     * <p>
     * Will be deprecated if/when OLCUT supports this.
     */
    public final static class TransformationList implements Configurable, Provenancable<ConfiguredObjectProvenance> {
        /**
         * The list of transformations.
         */
        @Config(description="A list of transformations to apply.")
        public List<Transformation> list;

        private TransformationList() {}

        /**
         * Constructs a transformation list.
         * @param list The list of transformations.
         */
        public TransformationList(List<Transformation> list) {
            this.list = list;
        }

        @Override
        public ConfiguredObjectProvenance getProvenance() {
            return new ConfiguredObjectProvenanceImpl(this,"TransformationList");
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof TransformationList)) return false;
            TransformationList that = (TransformationList) o;
            return list.equals(that.list);
        }

        @Override
        public int hashCode() {
            return Objects.hash(list);
        }
    }

}
