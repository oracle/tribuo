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
import org.tribuo.data.text.FeatureTransformer;
import org.tribuo.util.MurmurHash3;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Hashes the feature names to reduce the dimensionality.
 * <p>
 * Uses murmurhash3_x86_32 as the hashing function for the feature names.
 */
public class FeatureHasher implements FeatureTransformer {

    private static final Logger logger = Logger.getLogger(FeatureHasher.class.getName());

    @Config(mandatory = true,description="Dimension to map the hash into.")
    private int dimension;

    /**
     * Constructs a feature hasher using the supplied hash dimension.
     * @param dimension The dimension to reduce the hashed features into.
     */
    public FeatureHasher(int dimension) {
        this.dimension = dimension;
    }

    /**
     * For olcut.
     */
    private FeatureHasher() {}
    
    @Override
    public List<Feature> map(String tag, List<Feature> features) {

        List<Feature> hashedFeatures = new ArrayList<>();
        
        for (Feature feature : features) {
            int hash = MurmurHash3.murmurhash3_x86_32(feature.getName(), 0, feature.getName().length(), 38495);
            //int bit = hash & 1;
            int bit = MurmurHash3.murmurhash3_x86_32(feature.getName(), 0, feature.getName().length(), 77777) & 1; 
            hash = hash >>> 1;
            int code = hash % dimension;
                        
            int change = bit == 1 ? 1 : -1;

            Feature newFeature = new Feature(tag + "-hash="+code,change);
            hashedFeatures.add(newFeature);
        }

        return hashedFeatures;
        
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FeatureTransformer");
    }
}
