/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.provenance.impl;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceException;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import org.tribuo.FeatureSelector;
import org.tribuo.Output;
import org.tribuo.provenance.FeatureSelectorProvenance;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * An implementation of {@link FeatureSelectorProvenance} which delegates everything to
 * {@link SkeletalConfiguredObjectProvenance}. Used for feature selectors which don't
 * need to record any instance level information.
 */
public final class FeatureSelectorProvenanceImpl extends SkeletalConfiguredObjectProvenance implements FeatureSelectorProvenance {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a FeatureSelectorProvenanceImpl by reading the configured fields
     * from the host object.
     * @param host The object to record.
     * @param <T> The type of the feature selector.
     * @param <U> The output type of the feature selector.
     */
    public <U extends Output<U>, T extends FeatureSelector<U>> FeatureSelectorProvenanceImpl(T host) {
        super(host, "FeatureSelector");
    }

    /**
     * Construct a FeatureSelectorProvenanceImpl by extracting the necessary fields from the supplied
     * map.
     * @param map The serialised form of this provenance.
     */
    public FeatureSelectorProvenanceImpl(Map<String, Provenance> map) {
        super(extractProvenanceInfo(map));
    }

    /**
     * Extracts the class name and host short name provenances. Assumes the rest of the
     * map entries are configured parameters.
     * @param map The Map of provenance entries.
     * @return An extracted info object with the class name and host short name parsed out.
     */
    protected static ExtractedInfo extractProvenanceInfo(Map<String,Provenance> map) {
        String className;
        String hostTypeStringName;
        Map<String,Provenance> configuredParameters = new HashMap<>(map);
        if (configuredParameters.containsKey(ObjectProvenance.CLASS_NAME)) {
            className = configuredParameters.remove(ObjectProvenance.CLASS_NAME).toString();
        } else {
            throw new ProvenanceException("Failed to find class name when constructing FeatureSelectorProvenanceImpl");
        }
        if (configuredParameters.containsKey(HOST_SHORT_NAME)) {
            hostTypeStringName = configuredParameters.remove(HOST_SHORT_NAME).toString();
        } else {
            throw new ProvenanceException("Failed to find host type short name when constructing FeatureSelectorProvenanceImpl");
        }

        return new ExtractedInfo(className,hostTypeStringName,configuredParameters, Collections.emptyMap());
    }
}
