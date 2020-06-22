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

package org.tribuo.data.text;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Feature;

import java.util.List;

/**
 * A pipeline that takes a String and returns a List of {@link Feature}s.
 * This list is not guaranteed to have unique elements.
 */
public interface TextPipeline extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Extracts a list of features from the supplied text, using the tag to prepend the feature names.
     * @param tag The feature name tag.
     * @param data The text to extract.
     * @return The extracted features.
     */
    public List<Feature> process(String tag, String data);

}
