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

package org.tribuo.data.text;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Feature;

import java.util.List;

/**
 * A feature transformer maps a list of features to a new list of features.
 * Useful for example to apply the hashing trick to a set of features.
 * <p>
 * Note a list of features returned by a {@code FeatureTransformer} may contain
 * duplicate features, and should be reduced to ensure that each feature is unique.
 */
public interface FeatureTransformer extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Transforms features into a new list of features
     * @param tag The feature name tag.
     * @param features The features to transform.
     * @return The transformed features.
     */
    public List<Feature> map(String tag, List<Feature> features);
    
}
