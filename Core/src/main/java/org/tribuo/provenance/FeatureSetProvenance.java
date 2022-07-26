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

package org.tribuo.provenance;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Tribuo;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Contains provenance information for an instance of a {@link org.tribuo.SelectedFeatureSet}.
 * <p>
 * Made up of the class name of the feature set object, the provenance of
 * the training data, the provenance of the selection algorithm and the Tribuo version.
 */
public final class FeatureSetProvenance implements ObjectProvenance {
    private static final long serialVersionUID = 1L;

    private static final String DATASET = "dataset";
    private static final String SELECTOR = "feature-selector";
    private static final String TRIBUO_VERSION_STRING = "tribuo-version";

    private final String className;

    private final DatasetProvenance datasetProvenance;

    private final FeatureSelectorProvenance fsProvenance;

    private final String versionString;

    /**
     * Creates a feature set provenance tracking the dataset provenance and feature selector provenance.
     * @param className The class name of the host object (usually {@link org.tribuo.SelectedFeatureSet}).
     * @param datasetProvenance The dataset provenance.
     * @param fsProvenance The feature selector provenance.
     */
    public FeatureSetProvenance(String className, DatasetProvenance datasetProvenance,
                                FeatureSelectorProvenance fsProvenance) {
        this.className = className;
        this.datasetProvenance = datasetProvenance;
        this.fsProvenance = fsProvenance;
        this.versionString = Tribuo.VERSION;
    }

    /**
     * Used by the provenance unmarshalling system.
     * <p>
     * Throws {@link com.oracle.labs.mlrg.olcut.provenance.ProvenanceException} if there are missing
     * fields.
     * @param map The provenance map.
     */
    public FeatureSetProvenance(Map<String,Provenance> map) {
        this.className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME,StringProvenance.class, ModelProvenance.class.getSimpleName()).getValue();
        this.datasetProvenance = ObjectProvenance.checkAndExtractProvenance(map,DATASET,DatasetProvenance.class, FeatureSetProvenance.class.getSimpleName());
        this.fsProvenance = ObjectProvenance.checkAndExtractProvenance(map,SELECTOR,FeatureSelectorProvenance.class, FeatureSetProvenance.class.getSimpleName());
        this.versionString = ObjectProvenance.checkAndExtractProvenance(map,TRIBUO_VERSION_STRING,StringProvenance.class, FeatureSetProvenance.class.getSimpleName()).getValue();
    }

    /**
     * The training dataset provenance.
     * @return The training dataset provenance.
     */
    public DatasetProvenance getDatasetProvenance() {
        return datasetProvenance;
    }

    /**
     * The feature selector provenance.
     * @return The feature selector provenance.
     */
    public FeatureSelectorProvenance getFeatureSelectorProvenance() {
        return fsProvenance;
    }

    /**
     * The Tribuo version used to create this feature set.
     * @return The Tribuo version.
     */
    public String getTribuoVersion() {
        return versionString;
    }

    @Override
    public String toString() {
        return generateString("FeatureSet");
    }

    @Override
    public String getClassName() {
        return className;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FeatureSetProvenance)) return false;
        FeatureSetProvenance pairs = (FeatureSetProvenance) o;
        return className.equals(pairs.className) &&
               datasetProvenance.equals(pairs.datasetProvenance) &&
               fsProvenance.equals(pairs.fsProvenance) &&
               versionString.equals(pairs.versionString);
    }

    @Override
    public int hashCode() {
        return Objects.hash(className, datasetProvenance, versionString);
    }

    /**
     * Returns a list of all the provenances in this model provenance so subclasses
     * can append to the list.
     * @return A list of all the provenances in this class.
     */
    protected List<Pair<String,Provenance>> internalProvenances() {
        ArrayList<Pair<String,Provenance>> iterable = new ArrayList<>();
        iterable.add(new Pair<>(CLASS_NAME,new StringProvenance(CLASS_NAME,className)));
        iterable.add(new Pair<>(DATASET,datasetProvenance));
        iterable.add(new Pair<>(SELECTOR,fsProvenance));
        iterable.add(new Pair<>(TRIBUO_VERSION_STRING,new StringProvenance(TRIBUO_VERSION_STRING,versionString)));
        return iterable;
    }

    /**
     * Calls {@link #internalProvenances()} and returns the iterator from that list.
     * @return An iterator over all the provenances.
     */
    @Override
    public Iterator<Pair<String, Provenance>> iterator() {
        return internalProvenances().iterator();
    }
}
