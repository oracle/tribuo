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

package org.tribuo.provenance;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Tribuo;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;

/**
 * Provenance for evaluations.
 */
public final class EvaluationProvenance implements ObjectProvenance {
    private static final long serialVersionUID = 1L;

    private static final String MODEL_PROVENANCE_NAME = "model-provenance";
    private static final String DATASET_PROVENANCE_NAME = "dataset-provenance";
    private static final String TRIBUO_VERSION_STRING = "tribuo-version";

    private final StringProvenance className;
    private final ModelProvenance modelProvenance;
    private final DataProvenance datasetProvenance;
    private final StringProvenance versionString;

    /**
     * Constructs an evaluation provenance from the supplied provenances.
     * @param modelProvenance The model provenance.
     * @param datasetProvenance The test data provenance.
     */
    public EvaluationProvenance(ModelProvenance modelProvenance, DataProvenance datasetProvenance) {
        this.className = new StringProvenance(CLASS_NAME, EvaluationProvenance.class.getName());
        this.modelProvenance = modelProvenance;
        this.datasetProvenance = datasetProvenance;
        this.versionString = new StringProvenance(TRIBUO_VERSION_STRING,Tribuo.VERSION);
    }

    /**
     * Deserialization constructor.
     * @param map The provenances.
     */
    public EvaluationProvenance(Map<String,Provenance> map) {
        this.className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME,StringProvenance.class, EvaluationProvenance.class.getSimpleName());
        this.modelProvenance = ObjectProvenance.checkAndExtractProvenance(map,MODEL_PROVENANCE_NAME,ModelProvenance.class, EvaluationProvenance.class.getSimpleName());
        this.datasetProvenance = ObjectProvenance.checkAndExtractProvenance(map,DATASET_PROVENANCE_NAME,DataProvenance.class, EvaluationProvenance.class.getSimpleName());
        this.versionString = ObjectProvenance.checkAndExtractProvenance(map,TRIBUO_VERSION_STRING,StringProvenance.class, EvaluationProvenance.class.getSimpleName());
    }

    @Override
    public String getClassName() {
        return className.getValue();
    }

    /**
     * The test dataset provenance.
     * @return The test dataset provenance.
     */
    public DataProvenance getTestDatasetProvenance() {
        return datasetProvenance;
    }

    /**
     * The model provenance.
     * @return The model provenance.
     */
    public ModelProvenance getModelProvenance() {
        return modelProvenance;
    }

    /**
     * The Tribuo version used to create this dataset.
     * @return The Tribuo version.
     */
    public String getTribuoVersion() {
        return versionString.getValue();
    }

    @Override
    public Iterator<Pair<String, Provenance>> iterator() {
        ArrayList<Pair<String,Provenance>> list = new ArrayList<>();
        list.add(new Pair<>(CLASS_NAME, className));
        list.add(new Pair<>(MODEL_PROVENANCE_NAME, modelProvenance));
        list.add(new Pair<>(DATASET_PROVENANCE_NAME, datasetProvenance));
        list.add(new Pair<>(TRIBUO_VERSION_STRING,versionString));
        return list.iterator();
    }

    @Override
    public String toString() {
        return generateString("Evaluation");
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        EvaluationProvenance pairs = (EvaluationProvenance) o;
        return Objects.equals(className, pairs.className) &&
                Objects.equals(modelProvenance, pairs.modelProvenance) &&
                Objects.equals(datasetProvenance, pairs.datasetProvenance) &&
                Objects.equals(versionString, pairs.versionString);
    }

    @Override
    public int hashCode() {
        return Objects.hash(className, modelProvenance, datasetProvenance, versionString);
    }
}