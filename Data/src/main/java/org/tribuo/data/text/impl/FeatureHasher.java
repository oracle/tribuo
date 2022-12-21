/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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
import com.oracle.labs.mlrg.olcut.config.PropertyException;
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

    /**
     * Default value for the hash function seed.
     */
    public static final int DEFAULT_HASH_SEED = 38495;

    /**
     * Default value for the value hash function seed.
     */
    public static final int DEFAULT_VALUE_HASH_SEED = 77777;

    @Config(mandatory = true,description="Dimension to map the hash into.")
    private int dimension;

    @Config(description = "Seed used in the hash function.")
    private int hashSeed = DEFAULT_HASH_SEED;

    @Config(description = "Seed used for value hash function.")
    private int valueHashSeed = DEFAULT_VALUE_HASH_SEED;

    @Config(description = "Preserve input feature value.")
    private boolean preserveValue = false;

    /**
     * Constructs a feature hasher using the supplied hash dimension.
     * <p>
     * Note the hasher also hashes the feature value into {-1, 1}.
     * @param dimension The dimension to reduce the hashed features into.
     */
    public FeatureHasher(int dimension) {
        this(dimension, false);
    }

    /**
     * Constructs a feature hasher using the supplied hash dimension.
     * @param dimension The dimension to reduce the hashed features into.
     * @param preserveValue If true the feature value is used unaltered in the new features,
     *                      if false it is hashed into the values {-1, 1}.
     */
    public FeatureHasher(int dimension, boolean preserveValue) {
        this(dimension, DEFAULT_HASH_SEED, DEFAULT_VALUE_HASH_SEED, preserveValue);
    }

    /**
     * Constructs a feature hasher using the supplied hash dimension and seed values.
     * @param dimension The dimension to reduce the hashed features into.
     * @param hashSeed The seed used in the murmurhash computation.
     * @param valueHashSeed The seed used in the murmurhash computation for the feature value,
     *                      unused if {@code preserveValue} is true.
     * @param preserveValue If true the feature value is used unaltered in the new features,
     *                      if false it is hashed into the values {-1, 1}.
     */
    public FeatureHasher(int dimension, int hashSeed, int valueHashSeed, boolean preserveValue) {
        this.dimension = dimension;
        this.hashSeed = hashSeed;
        this.valueHashSeed = valueHashSeed;
        this.preserveValue = preserveValue;
        postConfig();
    }

    /**
     * For olcut.
     */
    private FeatureHasher() {}

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (dimension < 1) {
            throw new PropertyException("","dimension","Dimension must be positive, found " + dimension);
        }
    }
    
    @Override
    public List<Feature> map(String tag, List<Feature> features) {

        List<Feature> hashedFeatures = new ArrayList<>();
        
        for (Feature feature : features) {
            int hash = MurmurHash3.murmurhash3_x86_32(feature.getName(), 0, feature.getName().length(), hashSeed);
            hash = hash >>> 1;
            int code = hash % dimension;

            double value;
            if (preserveValue) {
                value = feature.getValue();
            } else {
                int bit = MurmurHash3.murmurhash3_x86_32(feature.getName(), 0, feature.getName().length(), valueHashSeed) & 1;
                value = bit == 1 ? 1 : -1;
            }

            Feature newFeature = new Feature(tag + "-hash="+code, value);
            hashedFeatures.add(newFeature);
        }

        return hashedFeatures;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FeatureTransformer");
    }
}
