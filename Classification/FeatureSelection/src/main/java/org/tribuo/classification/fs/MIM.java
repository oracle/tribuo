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

package org.tribuo.classification.fs;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.Dataset;
import org.tribuo.FeatureSelector;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.classification.Label;
import org.tribuo.provenance.FeatureSelectorProvenance;
import org.tribuo.provenance.impl.FeatureSelectorProvenanceImpl;

/**
 * Selects features according to their mutual information with the class label (aka Mutual Information Maximisation).
 * <p>
 * Uses equal width binning for the feature values.
 * <p>
 * See:
 * <pre>
 * Brown G, Pocock A, Zhao M-J, Lujan M.
 * "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection"
 * Journal of Machine Learning Research (JMLR) <a href="https://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf">PDF</a>.
 * </pre>
 */
public final class MIM implements FeatureSelector<Label> {

    @Config(mandatory = true, description = "Number of bins to use when discretising continuous features.")
    private int numBins;

    @Config(description = "Number of features to select, defaults to ranking all features.")
    private int k = SELECT_ALL;

    /**
     * For OLCUT.
     */
    private MIM() {}

    /**
     * Constructs a MIM feature selector that ranks all the features.
     * <p>
     * Continuous features are binned into {@code numBins} equal width bins.
     * @param numBins The number of bins, must be greater than 1.
     */
    public MIM(int numBins) {
        this(numBins,SELECT_ALL);
    }

    /**
     * Constructs a MIM feature selector that ranks the top {@code k} features.
     * <p>
     * Continuous features are binned into {@code numBins} equal width bins.
     * @param numBins The number of bins, must be greater than 1.
     * @param k The number of features to rank.
     */
    public MIM(int numBins, int k) {
        this.numBins = numBins;
        this.k = k;
        if ((k != SELECT_ALL) && (k < 1)) {
            throw new IllegalArgumentException("k must be -1 to select all features, or a positive number, found " + k);
        }
        if (numBins < 2) {
            throw new IllegalArgumentException("numBins must be >= 2, found " + numBins);
        }
    }

    @Override
    public boolean isOrdered() {
        return true;
    }

    @Override
    public SelectedFeatureSet select(Dataset<Label> dataset) {

    }

    @Override
    public FeatureSelectorProvenance getProvenance() {
        return new FeatureSelectorProvenanceImpl(this);
    }
}
