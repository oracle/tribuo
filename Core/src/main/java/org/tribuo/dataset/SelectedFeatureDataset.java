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

package org.tribuo.dataset;

import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.ImmutableDataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.MutableFeatureMap;
import org.tribuo.MutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.OutputInfo;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.VariableInfo;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.FeatureSetProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;

/**
 * This class creates a pruned dataset which only contains the selected features.
 * The new dataset may have fewer examples because if any of the examples
 * have no features after the minimum cardinality has been applied, then those
 * examples will not be added to the constructed dataset.
 * 
 * @param <T> The type of the outputs in this {@link Dataset}.
 */
public final class SelectedFeatureDataset<T extends Output<T>> extends ImmutableDataset<T> {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(SelectedFeatureDataset.class.getName());

    private final int k;

    private final SelectedFeatureSet featureSet;

    private final Set<String> selectedFeatures;

    private int numExamplesRemoved = 0;

    /**
     * Constructs a selected feature dataset using all the features in the supplied feature set.
     * @param dataset The dataset to copy.
     * @param featureSet The feature set to use.
     */
    public SelectedFeatureDataset(Dataset<T> dataset, SelectedFeatureSet featureSet) {
        this(dataset,featureSet,-1);
    }

    /**
     * Constructs a selected feature dataset.
     * @param dataset This dataset is left untouched and is used to populate the constructed dataset.
     * @param featureSet The feature set to use.
     * @param k Use the top k features if the feature set is ordered, or -1 to select all of them, throws {@link IllegalArgumentException} if it is unordered and set to a positive value.
     */
    public SelectedFeatureDataset(Dataset<T> dataset, SelectedFeatureSet featureSet, int k) {
        super(dataset.getProvenance(), dataset.getOutputFactory());
        this.featureSet = featureSet;
        this.k = k;

        // Validate feature set & k
        Set<String> tmpFeatures = new HashSet<>();
        if (k == 0 || featureSet.featureNames().size() == 0) {
            throw new IllegalArgumentException("Tried to select zero features.");
        } else if (k != -1 && !featureSet.isOrdered()) {
            throw new IllegalArgumentException("Tried to select the top " + k + " features from an unordered feature set.");
        } else if (k > featureSet.featureNames().size()) {
            throw new IllegalArgumentException("Tried to select more features than are available in feature set, requested " + k + ", found " + featureSet.featureNames().size());
        } else if (k > 0) {
            List<String> featureList = featureSet.featureNames();
            for (int i = 0; i < k; i++) {
                tmpFeatures.add(featureList.get(i));
            }
        } else if (k < -1) {
            throw new IllegalArgumentException("Supplied k " + k + " but only k == -1 or 1 < k < N} is allowed.");
        } else {
            tmpFeatures.addAll(featureSet.featureNames());
        }
        selectedFeatures = Collections.unmodifiableSet(tmpFeatures);

        // Check for feature set overlap with dataset
        FeatureMap wfm = dataset.getFeatureMap();
        Set<String> datasetFeatures = new HashSet<>(wfm.keySet());
        datasetFeatures.retainAll(selectedFeatures);
        if (datasetFeatures.size() == 0) {
            throw new IllegalArgumentException("The selected feature set had no overlap with the supplied dataset.");
        }

        // Generate feature subset examples, dropping any empty examples
        MutableFeatureMap featureMap = new MutableFeatureMap();
        MutableOutputInfo<T> outputInfo = dataset.getOutputFactory().generateInfo();
        List<Feature> features = new ArrayList<>();
        for (Example<T> ex : dataset) {
            features.clear();
            // Copy the example including the weight, output and metadata
            ArrayExample<T> copy = new ArrayExample<>(ex);
            for (Feature f : ex) {
                if (selectedFeatures.contains(f.getName())) {
                    // We'll definitely keep this example, so observe its feature values
                    featureMap.add(f.getName(), f.getValue());
                } else {
                    features.add(f);
                }
            }
            // Check if there are features to remove
            if (features.size() > 0) {
                copy.removeFeatures(features);
            }
            // If the example is still valid then add it, otherwise increment the removed counter
            if (copy.size() > 0) {
                data.add(copy);
                outputInfo.observe(ex.getOutput());
            } else {
                numExamplesRemoved++;
            }
        }

        // Rebuild feature and output maps
        this.featureIDMap = new ImmutableFeatureMap(featureMap);
        this.outputIDInfo = outputInfo.generateImmutableOutputInfo();

        if(numExamplesRemoved > 0) {
            logger.info(String.format("filtered out %d examples because it had zero features after the selected feature set was applied.", numExamplesRemoved));
        }
    }

    /**
     * The number of examples removed due to a lack of features.
     * @return The number of removed examples.
     */
    public int getNumExamplesRemoved() {
        return numExamplesRemoved;
    }

    /**
     * The number of features to use.
     * <p>
     * -1 signals that all features in the supplied feature set were used.
     * @return The number of features to use.
     */
    public int getK() {
        return k;
    }

    /**
     * The feature set.
     * @return The feature set.
     */
    public SelectedFeatureSet getFeatureSet() {
        return featureSet;
    }

    /**
     * The subset of the selected feature set.
     * @return The used subset of the selected feature set.
     */
    public Set<String> getSelectedFeatures() {
        return selectedFeatures;
    }

    @Override
    public DatasetProvenance getProvenance() {
        return new SelectedFeatureDatasetProvenance(this);
    }

    /**
     * Provenance for {@link SelectedFeatureDataset}.
     */
    public static final class SelectedFeatureDatasetProvenance extends DatasetProvenance {
        private static final long serialVersionUID = 1L;

        private static final String K = "k";
        private static final String FEATURE_SET_PROVENANCE = "feature-set-provenance";
        private static final String DATASET_PROVENANCE = "original-data-provenance";

        private final IntProvenance k;
        private final FeatureSetProvenance featureSetProvenance;
        private final DataProvenance datasetProvenance;

        <T extends Output<T>> SelectedFeatureDatasetProvenance(SelectedFeatureDataset<T> dataset) {
            super(dataset.sourceProvenance, new ListProvenance<>(), dataset);
            this.k = new IntProvenance(K,dataset.k);
            this.featureSetProvenance = dataset.featureSet.getProvenance();
            this.datasetProvenance = dataset.sourceProvenance;
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public SelectedFeatureDatasetProvenance(Map<String,Provenance> map) {
            super(map);
            this.k = ObjectProvenance.checkAndExtractProvenance(map,K,IntProvenance.class,SelectedFeatureDatasetProvenance.class.getSimpleName());
            this.featureSetProvenance = ObjectProvenance.checkAndExtractProvenance(map,FEATURE_SET_PROVENANCE, FeatureSetProvenance.class, SelectedFeatureDatasetProvenance.class.getSimpleName());
            this.datasetProvenance = ObjectProvenance.checkAndExtractProvenance(map,DATASET_PROVENANCE, DataProvenance.class, SelectedFeatureDatasetProvenance.class.getSimpleName());
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            if (!super.equals(o)) return false;
            SelectedFeatureDatasetProvenance pairs = (SelectedFeatureDatasetProvenance) o;
            return k.equals(pairs.k) && featureSetProvenance.equals(pairs.featureSetProvenance) && datasetProvenance.equals(pairs.datasetProvenance);
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), k, featureSetProvenance, datasetProvenance);
        }

        @Override
        protected List<Pair<String, Provenance>> allProvenances() {
            List<Pair<String,Provenance>> provenances = super.allProvenances();
            provenances.add(new Pair<>(K,k));
            provenances.add(new Pair<>(FEATURE_SET_PROVENANCE,featureSetProvenance));
            provenances.add(new Pair<>(DATASET_PROVENANCE,datasetProvenance));
            return provenances;
        }
    }
}
