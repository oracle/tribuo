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
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.util.SortUtil;
import org.tribuo.Dataset;
import org.tribuo.FeatureSelector;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.classification.Label;
import org.tribuo.provenance.FeatureSelectorProvenance;
import org.tribuo.provenance.FeatureSetProvenance;
import org.tribuo.provenance.impl.FeatureSelectorProvenanceImpl;

import java.util.ArrayList;

/**
 * Selects features according to their mutual information with the class label (aka Mutual Information Maximisation).
 * <p>
 * Uses equal width binning for the feature values.
 * <p>
 * See:
 * <pre>
 * Brown G, Pocock A, Zhao M-J, Lujan M.
 * "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection"
 * Journal of Machine Learning Research (JMLR), 2012, <a href="https://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf">PDF</a>.
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
        this(SELECT_ALL, numBins);
    }

    /**
     * Constructs a MIM feature selector that ranks the top {@code k} features.
     * <p>
     * Continuous features are binned into {@code numBins} equal width bins.
     * @param k The number of features to rank.
     * @param numBins The number of bins, must be greater than 1.
     */
    public MIM(int k, int numBins) {
        this.numBins = numBins;
        this.k = k;
        if ((k != SELECT_ALL) && (k < 1)) {
            throw new IllegalArgumentException("k must be -1 to select all features, or a positive number, found " + k);
        }
        if (numBins < 2) {
            throw new IllegalArgumentException("numBins must be >= 2, found " + numBins);
        }
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if ((k != SELECT_ALL) && (k < 1)) {
            throw new PropertyException("","k","k must be -1 to select all features, or a positive number, found " + k);
        }
        if (numBins < 2) {
            throw new PropertyException("","numBins","numBins must be >= 2, found " + numBins);
        }
    }

    @Override
    public boolean isOrdered() {
        return true;
    }

    @Override
    public SelectedFeatureSet select(Dataset<Label> dataset) {
        FSMatrix data = FSMatrix.buildMatrix(dataset,numBins);

        ImmutableFeatureMap fmap = data.getFeatureMap();

        int max = k == -1 ? fmap.size() : Math.min(k,fmap.size());

        double[] mi = data.miList();

        int[] sortOrder = SortUtil.argsort(mi,false);

        ArrayList<String> names = new ArrayList<>();
        ArrayList<Double> scores = new ArrayList<>();
        for (int i = 0; i < max; i++) {
            int id = sortOrder[i];
            names.add(fmap.get(id).getName());
            scores.add(mi[id]);
        }

        FeatureSetProvenance provenance = new FeatureSetProvenance(SelectedFeatureSet.class.getName(),dataset.getProvenance(),getProvenance());
        return new SelectedFeatureSet(names,scores,isOrdered(),provenance);
    }

    @Override
    public FeatureSelectorProvenance getProvenance() {
        return new FeatureSelectorProvenanceImpl(this);
    }
}
