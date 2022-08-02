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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import java.io.Serializable;
import org.tribuo.provenance.FeatureSetProvenance;

import java.util.Collections;
import java.util.List;

/**
 * A record-like class for a selected feature set.
 * <p>
 * Uses record style accessors as it may be refactored into a record one day.
 */
public final class SelectedFeatureSet implements Provenancable<FeatureSetProvenance>, Serializable {
    private static final long serialVersionUID = 1L;

    private final List<String> featureNames;

    private final List<Double> featureScores;

    private final FeatureSetProvenance provenance;

    private final boolean isOrdered;

    /**
     * Create a selected feature set.
     * @param featureNames The feature names.
     * @param featureScores The feature scores.
     * @param isOrdered Is this feature set ordered?
     * @param provenance The provenance of the feature selection.
     */
    public SelectedFeatureSet(List<String> featureNames, List<Double> featureScores, boolean isOrdered, FeatureSetProvenance provenance) {
        this.featureNames = Collections.unmodifiableList(featureNames);
        this.featureScores = Collections.unmodifiableList(featureScores);
        this.isOrdered = isOrdered;
        this.provenance = provenance;
    }

    /**
     * The selected feature names in a possibly ordered list.
     * @return The selected feature names.
     */
    public List<String> featureNames() {
        return featureNames;
    }

    /**
     * The selected feature scores in a possibly ordered list.
     * <p>
     * If the algorithm did not produce scores then these values are all {@link Double#NaN}.
     * @return The selected feature scores.
     */
    public List<Double> featureScores() {
        return featureScores;
    }

    /**
     * The provenance of the feature set.
     * @return The feature set provenance.
     */
    public FeatureSetProvenance provenance() {
        return provenance;
    }

    @Override
    public FeatureSetProvenance getProvenance() {
        return provenance;
    }

    /**
     * Is this feature set ordered?
     * @return True if the set is ordered.
     */
    public boolean isOrdered() {
        return isOrdered;
    }
}
